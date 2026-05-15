# Copyright Sierra

import json
import re
from typing import Any, Dict, List, Optional, TypedDict

from langgraph.graph import StateGraph, END
from litellm import completion

from tau_bench.agents.base import Agent
from tau_bench.envs.base import Env
from tau_bench.model_utils.model.utils import approx_num_tokens
from tau_bench.types import Action, SolveResult, RESPOND_ACTION_NAME


class PairState(TypedDict):
    conversation: List[Dict[str, Any]]
    messages: List[Dict[str, Any]]
    user_request: str
    plan_text: str
    plan_steps: List[Dict[str, Any]]
    plan_feedback: str
    validation_feedback: str
    audit_feedback: str
    needs_replan: bool
    replan_attempts: int
    tool_outputs: List[Dict[str, Any]]
    last_user_message: str
    total_cost: float
    done: bool
    skip_respond: bool
    reward: float
    info: Dict[str, Any]
    env_step_count: int
    max_steps: int


class PairAgent(Agent):
    def __init__(
        self,
        tools_info: List[Dict[str, Any]],
        wiki: str,
        rules: List[str],
        model: str,
        provider: str,
        temperature: float = 0.0,
        context_budget: int = 32000,
    ) -> None:
        self.tools_info = tools_info
        self.tool_schema_map = {
            tool["function"]["name"]: tool for tool in tools_info
        }
        self.wiki = wiki
        self.rules = rules
        self.model = model
        self.provider = provider
        self.temperature = temperature
        self.context_budget = context_budget

    def solve(
        self, env: Env, task_index: Optional[int] = None, max_num_steps: int = 30
    ) -> SolveResult:
        env_reset_res = env.reset(task_index=task_index)
        obs = env_reset_res.observation
        info = env_reset_res.info.model_dump()
        reward = 0.0
        state: PairState = {
            "conversation": [
                {"role": "system", "content": self.wiki},
                {"role": "user", "content": obs},
            ],
            "messages": [
                {"role": "system", "content": self.wiki},
                {"role": "user", "content": obs},
            ],
            "user_request": obs,
            "plan_text": "",
            "plan_steps": [],
            "plan_feedback": "",
            "validation_feedback": "",
            "audit_feedback": "",
            "needs_replan": False,
            "replan_attempts": 0,
            "tool_outputs": [],
            "last_user_message": obs,
            "total_cost": 0.0,
            "done": False,
            "skip_respond": False,
            "reward": reward,
            "info": info,
            "env_step_count": 0,
            "max_steps": max_num_steps,
        }

        graph = self._build_graph(env)

        while state["env_step_count"] < max_num_steps and not state["done"]:
            state = graph.invoke(state)

            if state["needs_replan"]:
                if state["validation_feedback"]:
                    state["plan_feedback"] = state["validation_feedback"]
                    state["validation_feedback"] = ""
                if state["audit_feedback"]:
                    state["plan_feedback"] = state["audit_feedback"]
                    state["audit_feedback"] = ""
                if state["skip_respond"]:
                    state["skip_respond"] = False
                    continue
                if state["replan_attempts"] < 1:
                    state["replan_attempts"] += 1
                    continue

            if state["done"]:
                break

            # Prepare for a new turn if the user hasn't stopped.
            state["plan_text"] = ""
            state["plan_steps"] = []
            state["plan_feedback"] = ""
            state["validation_feedback"] = ""
            state["audit_feedback"] = ""
            state["needs_replan"] = False
            state["replan_attempts"] = 0
            state["skip_respond"] = False
            state["user_request"] = state["last_user_message"]

        return SolveResult(
            reward=state["reward"],
            info=state["info"],
            messages=state["messages"],
            total_cost=state["total_cost"],
        )

    def _build_graph(self, env: Env):
        graph = StateGraph(PairState)
        graph.add_node("compress", lambda s: self._compress(s))
        graph.add_node("plan", lambda s: self._plan(s))
        graph.add_node("validate", lambda s: self._validate(s))
        graph.add_node("execute", lambda s: self._execute(s, env))
        graph.add_node("critic", lambda s: self._critic(s))
        graph.add_node("respond", lambda s: self._respond(s, env))

        graph.set_entry_point("compress")
        graph.add_edge("compress", "plan")
        graph.add_edge("plan", "validate")
        graph.add_edge("validate", "execute")
        graph.add_edge("execute", "critic")
        graph.add_edge("critic", "respond")
        graph.add_edge("respond", END)
        return graph.compile()

    def _compress(self, state: PairState) -> PairState:
        conversation_text = self._format_conversation(state["conversation"])
        if approx_num_tokens(conversation_text) <= int(self.context_budget * 0.6):
            return state

        keep_last = 6
        prefix = state["conversation"][:-keep_last]
        suffix = state["conversation"][-keep_last:]
        if not prefix:
            return state

        summary_prompt = (
            "Summarize the following conversation segment in <=200 tokens, "
            "preserving: user identity, current reservation/order IDs, any confirmed "
            "decisions, and pending questions."
        )
        summary_input = self._format_conversation(prefix)
        summary, cost = self._call_llm(summary_prompt, summary_input)
        state["total_cost"] += cost
        state["conversation"] = (
            [{"role": "system", "content": f"Summary so far: {summary}"}] + suffix
        )
        return state

    def _plan(self, state: PairState) -> PairState:
        tool_names = ", ".join(sorted(self.tool_schema_map.keys()))
        known = self._extract_known_identifiers(state["conversation"], state["tool_outputs"])
        tool_descriptions = []
        for name in sorted(self.tool_schema_map.keys()):
            schema = self.tool_schema_map[name]
            desc = schema.get("function", {}).get("description", "")
            params = schema.get("function", {}).get("parameters", {})
            param_names = list(params.get("properties", {}).keys())
            tool_descriptions.append(f"- {name}: {desc} (params: {', '.join(param_names)})")
        tools_block = "\n".join(tool_descriptions)
        plan_prompt = (
            "You are a task planner for a customer-service agent.\n"
            "Given the conversation so far and the user's request, produce a PLAN "
            "as a numbered list of steps. Each step is one of:\n"
            "- LOOKUP(tool_name, description of what to retrieve)\n"
            "- ACTION(tool_name, description of what to do)\n"
            "- ASK(what to ask the user)\n"
            "- CONFIRM(what to confirm before write action)\n"
            "Do NOT include specific argument values - only the intent.\n"
            f"Available tools:\n{tools_block}\n\n"
            "IMPORTANT: Look at the Known identifiers below. If an identifier like user_id "
            "is already known, do NOT ask for it again. Instead, use the appropriate LOOKUP "
            "tool with that identifier.\n"
            "If a required identifier is missing from Known identifiers AND cannot be found "
            "in the conversation, first plan a LOOKUP to retrieve it "
            "(for example, use get_user_details to find reservation IDs) before asking the user. "
            "Do not ask repeatedly for the same identifier if it can be retrieved from tools "
            "or is already provided in the conversation."
        )
        conversation_text = self._format_conversation(state["conversation"])
        feedback = state["plan_feedback"]
        if state["validation_feedback"]:
            feedback = state["validation_feedback"]
        if state["audit_feedback"]:
            feedback = state["audit_feedback"]

        user_prompt = (
            f"Conversation:\n{conversation_text}\n\n"
            f"User request:\n{state['user_request']}\n\n"
            f"Known identifiers:\n{json.dumps(known)}\n\n"
            f"Feedback (if any):\n{feedback}\n\n"
            "PLAN:"
        )
        plan_text, cost = self._call_llm(plan_prompt, user_prompt)
        state["total_cost"] += cost
        state["plan_text"] = plan_text.strip()
        state["plan_steps"] = self._parse_plan_steps(plan_text, known)
        state["needs_replan"] = False
        return state

    def _validate(self, state: PairState) -> PairState:
        rules_text = "\n".join(self.rules) if self.rules else "(No explicit rules)"
        tool_names = ", ".join(sorted(self.tool_schema_map.keys()))
        known = self._extract_known_identifiers(state["conversation"], state["tool_outputs"])
        policy_prompt = (
            "You are a domain-policy compliance checker.\n"
            "Given the PLAN and the current state (user profile, reservation details), "
            "check each step against the following rules:\n"
            f"{rules_text}\n\n"
            "Also use the domain wiki for policy.\n"
            f"Allowed tool names: {tool_names}\n"
            "For each step, respond:\n"
            "VALID - rule X supports this\n"
            "INVALID - violates rule Y; suggest alternative"
        )
        tool_summary = json.dumps(state["tool_outputs"], indent=2)
        user_prompt = (
            f"PLAN:\n{state['plan_text']}\n\n"
            f"Current tool outputs:\n{tool_summary}\n\n"
            f"Domain wiki:\n{self.wiki}\n"
        )
        validation, cost = self._call_llm(policy_prompt, user_prompt)
        state["total_cost"] += cost
        unknown_tools = []
        for step in state["plan_steps"]:
            if step.get("type") in ("LOOKUP", "ACTION"):
                tool_name = step.get("tool", "")
                if tool_name not in self.tool_schema_map:
                    unknown_tools.append(tool_name)
        lookup_feedback = ""
        has_user_lookup = any(
            step.get("type") == "LOOKUP" and step.get("tool") == "get_user_details"
            for step in state["plan_steps"]
        )
        asks_reservation = any(
            step.get("type") == "ASK"
            and "reservation" in (step.get("description") or "").lower()
            for step in state["plan_steps"]
        )
        asks_email = any(
            step.get("type") == "ASK"
            and "email" in (step.get("description") or "").lower()
            for step in state["plan_steps"]
        )
        asks_order = any(
            step.get("type") == "ASK"
            and "order" in (step.get("description") or "").lower()
            for step in state["plan_steps"]
        )
        asks_user_id = any(
            step.get("type") == "ASK"
            and "user id" in (step.get("description") or "").lower()
            for step in state["plan_steps"]
        )
        if asks_reservation and "get_user_details" in self.tool_schema_map and not has_user_lookup:
            lookup_feedback = (
                "INVALID - reservation id asked without first using get_user_details."
            )
        if asks_reservation and known.get("reservation_id"):
            lookup_feedback = "\n".join(
                item
                for item in [
                    lookup_feedback,
                    "INVALID - reservation id asked even though it is already provided.",
                ]
                if item
            )
        if asks_email and "get_user_details" in self.tool_schema_map and not has_user_lookup:
            lookup_feedback = "\n".join(
                item
                for item in [
                    lookup_feedback,
                    "INVALID - email asked without first using get_user_details.",
                ]
                if item
            )
        if asks_email and known.get("email"):
            lookup_feedback = "\n".join(
                item
                for item in [
                    lookup_feedback,
                    "INVALID - email asked even though it is already provided.",
                ]
                if item
            )
        if asks_order and "get_user_details" in self.tool_schema_map and not has_user_lookup:
            lookup_feedback = "\n".join(
                item
                for item in [
                    lookup_feedback,
                    "INVALID - order id asked without first using get_user_details.",
                ]
                if item
            )
        if asks_user_id and known.get("user_id"):
            lookup_feedback = "\n".join(
                item
                for item in [
                    lookup_feedback,
                    "INVALID - user id asked even though it is already provided.",
                ]
                if item
            )
        elif asks_user_id and "get_user_details" in self.tool_schema_map and not has_user_lookup:
            lookup_feedback = "\n".join(
                item
                for item in [
                    lookup_feedback,
                    "INVALID - user id asked without first using get_user_details when it can be derived from context.",
                ]
                if item
            )
        ask_descriptions = [
            (step.get("description") or "").lower()
            for step in state["plan_steps"]
            if step.get("type") == "ASK"
        ]
        if ask_descriptions:
            repeats = {}
            for desc in ask_descriptions:
                key = ""
                if "email" in desc:
                    key = "email"
                elif "reservation" in desc:
                    key = "reservation"
                elif "order" in desc:
                    key = "order"
                elif "user id" in desc:
                    key = "user id"
                if key:
                    repeats[key] = repeats.get(key, 0) + 1
            repeated_keys = [k for k, v in repeats.items() if v > 1]
            if repeated_keys:
                lookup_feedback = "\n".join(
                    item
                    for item in [
                        lookup_feedback,
                        "INVALID - repeated ASK steps for: " + ", ".join(repeated_keys) + ".",
                    ]
                    if item
                )
        unknown_feedback = ""
        if unknown_tools:
            unknown_feedback = (
                "INVALID - plan references unknown tools: "
                + ", ".join(sorted(set(unknown_tools)))
            )
        state["validation_feedback"] = "\n".join(
            item for item in [validation, unknown_feedback, lookup_feedback] if item
        )
        state["needs_replan"] = "INVALID" in state["validation_feedback"].upper()
        return state

    def _execute(self, state: PairState, env: Env) -> PairState:
        state["skip_respond"] = False
        for step in state["plan_steps"]:
            if state["env_step_count"] >= state["max_steps"] or state["done"]:
                break
            step_type = step.get("type")
            if step_type in ("LOOKUP", "ACTION"):
                tool_name = step.get("tool")
                if not tool_name:
                    continue
                if step_type == "ACTION" and not self._confirmation_gate(
                    tool_name, state["conversation"]
                ):
                    question = f"Please confirm: {step.get('description', '')}"
                    state = self._ask_user(state, env, question)
                    state["skip_respond"] = True
                    if not self._is_affirmative(state["last_user_message"]):
                        state["plan_feedback"] = (
                            "User did not confirm the requested change."
                        )
                        state["needs_replan"] = True
                    break

                args = self._build_tool_args(step, state)
                action = Action(name=tool_name, kwargs=args)
                env_response = env.step(action)
                state["env_step_count"] += 1
                state["reward"] = env_response.reward
                state["info"] = {**state["info"], **env_response.info.model_dump()}
                state["tool_outputs"].append(
                    {
                        "tool": tool_name,
                        "arguments": args,
                        "output": env_response.observation,
                    }
                )
                state["messages"].extend(
                    [
                        {
                            "role": "assistant",
                            "content": f"Calling {tool_name} with {json.dumps(args)}",
                        },
                        {
                            "role": "tool",
                            "name": tool_name,
                            "content": env_response.observation,
                        },
                    ]
                )
                state["conversation"].extend(
                    [
                        {
                            "role": "assistant",
                            "content": f"Tool call: {tool_name}",
                        },
                        {
                            "role": "tool",
                            "content": env_response.observation,
                        },
                    ]
                )
                if env_response.done:
                    state["done"] = True
                    break
            elif step_type in ("ASK", "CONFIRM"):
                raw_description = step.get("description", "")
                question = self._reformulate_as_question(raw_description, state)
                state = self._ask_user(state, env, question)
                state["skip_respond"] = True
                state["needs_replan"] = True
                break
        return state

    def _critic(self, state: PairState) -> PairState:
        if state["done"] or state["skip_respond"]:
            return state

        critic_prompt = (
            "You are a post-execution auditor.\n"
            "Given: the user's original request, the plan that was executed, "
            "the tool call results, and the current state, determine whether "
            "the outcome satisfies the user's request.\n"
            "Respond with one of:\n"
            "SATISFIED - goal achieved\n"
            "PARTIAL - {what is missing}\n"
            "WRONG - {what went wrong, suggest re-plan}"
        )
        tool_summary = json.dumps(state["tool_outputs"], indent=2)
        user_prompt = (
            f"User request:\n{state['user_request']}\n\n"
            f"Plan:\n{state['plan_text']}\n\n"
            f"Tool results:\n{tool_summary}\n"
        )
        audit, cost = self._call_llm(critic_prompt, user_prompt)
        state["total_cost"] += cost
        state["audit_feedback"] = audit
        audit_upper = audit.upper()
        if "WRONG" in audit_upper or "PARTIAL" in audit_upper:
            state["needs_replan"] = True
        return state

    def _respond(self, state: PairState, env: Env) -> PairState:
        if state["skip_respond"] or state["done"]:
            return state

        responder_prompt = (
            "You are a customer service agent speaking DIRECTLY to the customer.\n"
            "Write a complete, natural message as if you are talking to them face to face.\n\n"
            "RULES:\n"
            "- Write a COMPLETE sentence or paragraph, NOT a fragment, topic, or summary.\n"
            "- BAD example: 'cancellation eligibility based on created time and policy rules'\n"
            "- GOOD example: 'I checked your reservation and it is eligible for cancellation since it was created over 24 hours ago. Would you like me to proceed?'\n"
            "- Address the customer directly using 'you/your'.\n"
            "- If you need more information, ask a specific question.\n"
            "- Only use data from the tool results below. Do NOT fabricate any information.\n"
            "- If tool results don't contain the needed data, tell the customer you need to look it up."
        )
        conversation_text = self._format_conversation(state["conversation"])
        tool_summary = json.dumps(state["tool_outputs"], indent=2)
        user_prompt = (
            f"Conversation:\n{conversation_text}\n\n"
            f"Tool results:\n{tool_summary}\n\n"
            "Write your response to the customer:"
        )
        response_text, cost = self._call_llm(responder_prompt, user_prompt)
        state["total_cost"] += cost
        action = Action(name=RESPOND_ACTION_NAME, kwargs={"content": response_text})
        env_response = env.step(action)
        state["env_step_count"] += 1
        state["reward"] = env_response.reward
        state["info"] = {**state["info"], **env_response.info.model_dump()}
        user_obs = self._sanitize_user_observation(env_response.observation)
        state["messages"].extend(
            [
                {"role": "assistant", "content": response_text},
                {"role": "user", "content": user_obs},
            ]
        )
        state["conversation"].extend(
            [
                {"role": "assistant", "content": response_text},
                {"role": "user", "content": user_obs},
            ]
        )
        state["last_user_message"] = user_obs
        if env_response.done:
            state["done"] = True
        return state

    def _call_llm(self, system_prompt: str, user_prompt: str):
        res = completion(
            model=self.model,
            custom_llm_provider=self.provider,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=self.temperature,
        )
        message = res.choices[0].message
        content = self._strip_think_tags(message.content or "")
        return content, res._hidden_params["response_cost"] or 0

    @staticmethod
    def _strip_think_tags(text: str) -> str:
        """Strip <think>...</think> blocks produced by reasoning models like Qwen3."""
        return re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL).strip()

    def _format_conversation(self, messages: List[Dict[str, Any]]) -> str:
        lines = []
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            lines.append(f"{role}: {content}")
        return "\n".join(lines)

    def _parse_plan_steps(self, plan_text: str, known: Optional[Dict[str, str]] = None) -> List[Dict[str, Any]]:
        steps = []
        for line in plan_text.splitlines():
            line = line.strip()
            if not line:
                continue
            # Try strict format first: "1. LOOKUP(tool, desc)" or "- LOOKUP(tool, desc)"
            match = re.match(r"^(?:[\-\*]\s+)?(?:\d+[\.\)\-]\s*)?(?:\*\*)?\s*(LOOKUP|ACTION|ASK|CONFIRM)(?:\*\*)?\s*\((.+?)\)(?:\s.*)?$", line)
            if not match:
                # Try to match lines like "- LOOKUP: tool, desc" or "LOOKUP tool, desc"
                match = re.match(r"^(?:[\-\*]\s*)?(?:\*\*)?\s*(LOOKUP|ACTION|ASK|CONFIRM)[:\s]+(.+?)\s*$", line)
            if not match:
                continue
            step_type = match.group(1)
            inside = match.group(2).strip()
            if step_type in ("LOOKUP", "ACTION"):
                tool_name, description = self._split_first_comma(inside)
                steps.append(
                    {
                        "type": step_type,
                        "tool": tool_name,
                        "description": description,
                    }
                )
            else:
                steps.append({"type": step_type, "description": inside})
        if not steps:
            # Instead of always asking a clarifying question (which causes loops),
            # try to do a lookup if we have a user_id
            if known and known.get("user_id") and "get_user_details" in self.tool_schema_map:
                steps.append({
                    "type": "LOOKUP",
                    "tool": "get_user_details",
                    "description": "Retrieve user details to understand the request",
                })
            else:
                steps.append({"type": "ASK", "description": "Could you please provide your user ID so I can assist you?"})
        return steps

    def _split_first_comma(self, text: str) -> tuple[str, str]:
        parts = [p.strip() for p in text.split(",", 1)]
        if len(parts) == 1:
            return parts[0], ""
        return parts[0], parts[1]

    def _build_tool_args(self, step: Dict[str, Any], state: PairState) -> Dict[str, Any]:
        tool_name = step.get("tool", "")
        schema = self.tool_schema_map.get(tool_name, {})
        if not schema:
            raise ValueError(f"Unknown tool schema for {tool_name}")
        conversation_text = self._format_conversation(state["conversation"])
        known = self._extract_known_identifiers(state["conversation"], state["tool_outputs"])
        assembler_prompt = (
            "You are a tool-call argument builder.\n"
            "Given: the validated plan step, the conversation history, known identifiers, "
            "the current data state (tool outputs), "
            "and the tool schema, produce the exact JSON arguments for this tool call.\n"
            "IMPORTANT: Extract argument values from the conversation, known identifiers, "
            "and tool outputs ONLY. "
            "For example, if the user said 'my user id is abc123', use 'abc123' as the user_id argument.\n"
            "NEVER fabricate or make up values such as flight numbers, reservation IDs, or dates "
            "that are not explicitly present in the conversation or tool outputs.\n"
            "For multi-item operations, list ALL items explicitly.\n"
            "Double-check each argument against the schema before responding.\n"
            "Return only JSON."
        )
        tool_summary = json.dumps(state["tool_outputs"], indent=2)
        user_prompt = (
            f"Plan step: {json.dumps(step)}\n\n"
            f"Conversation:\n{conversation_text}\n\n"
            f"Known identifiers: {json.dumps(known)}\n\n"
            f"Tool outputs: {tool_summary}\n\n"
            f"Tool schema: {json.dumps(schema)}"
        )
        for _ in range(2):
            res = completion(
                model=self.model,
                custom_llm_provider=self.provider,
                messages=[
                    {"role": "system", "content": assembler_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                tools=[schema],
                tool_choice={"type": "function", "function": {"name": tool_name}},
                temperature=self.temperature,
            )
            state["total_cost"] += res._hidden_params.get("response_cost", 0.0) or 0.0
            message = res.choices[0].message.model_dump()
            tool_calls = message.get("tool_calls") or []
            if tool_calls and tool_calls[0].get("function"):
                return json.loads(tool_calls[0]["function"]["arguments"])
            try:
                return self._safe_json_loads(message.get("content") or "")
            except json.JSONDecodeError:
                continue
        raise ValueError("Assembler did not return valid JSON arguments")

    def _safe_json_loads(self, text: str) -> Dict[str, Any]:
        cleaned = text.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```[a-zA-Z]*", "", cleaned)
            cleaned = cleaned.strip("`\n")
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1:
            cleaned = cleaned[start : end + 1]
        return json.loads(cleaned)

    def _reformulate_as_question(self, raw_description: str, state: PairState) -> str:
        """Convert a raw plan step description into a proper user-facing question.

        Plan descriptions like 'whether the user wants to proceed with X' are
        internal notes, not proper messages. This reformulates them into natural
        conversational questions that won't confuse the user model.
        """
        conversation_text = self._format_conversation(state["conversation"][-4:])
        prompt = (
            "Convert the following internal note into a proper customer-facing message. "
            "Write as a customer service agent speaking directly to the customer.\n\n"
            "RULES:\n"
            "- Write a COMPLETE, natural sentence or question.\n"
            "- Address the customer directly with 'you/your'.\n"
            "- BAD: 'whether the user wants to proceed with cancellation'\n"
            "- GOOD: 'Would you like me to proceed with the cancellation?'\n"
            "- BAD: 'ask for reservation ID'\n"
            "- GOOD: 'Could you please provide your reservation ID?'\n"
            "- Output ONLY the reformulated message, nothing else."
        )
        user_prompt = (
            f"Recent conversation:\n{conversation_text}\n\n"
            f"Internal note to reformulate:\n{raw_description}"
        )
        question, cost = self._call_llm(prompt, user_prompt)
        state["total_cost"] += cost
        # Fallback: if reformulation fails or is empty, use the raw description
        return question.strip() if question.strip() else raw_description

    def _ask_user(self, state: PairState, env: Env, question: str) -> PairState:
        action = Action(name=RESPOND_ACTION_NAME, kwargs={"content": question})
        env_response = env.step(action)
        state["env_step_count"] += 1
        state["reward"] = env_response.reward
        state["info"] = {**state["info"], **env_response.info.model_dump()}
        user_obs = self._sanitize_user_observation(env_response.observation)
        state["messages"].extend(
            [
                {"role": "assistant", "content": question},
                {"role": "user", "content": user_obs},
            ]
        )
        state["conversation"].extend(
            [
                {"role": "assistant", "content": question},
                {"role": "user", "content": user_obs},
            ]
        )
        state["last_user_message"] = user_obs
        if env_response.done:
            state["done"] = True
        return state

    def _sanitize_user_observation(self, obs: str) -> str:
        """Clean user model output before storing in conversation.

        Strips <think> blocks (Qwen3 reasoning) and removes any tool-call-like
        content that the user model may hallucinate when role-confused.
        """
        cleaned = self._strip_think_tags(obs)
        # Strip markdown code blocks containing tool calls (user model role confusion)
        cleaned = re.sub(
            r"```(?:tool_code|python|json)?\s*\n?.*?```",
            "",
            cleaned,
            flags=re.DOTALL,
        ).strip()
        return cleaned if cleaned else obs

    def _confirmation_gate(self, tool_name: str, conversation: List[Dict[str, Any]]) -> bool:
        write_tools = {
            "cancel_reservation",
            "book_reservation",
            "update_reservation_flights",
            "update_reservation_passengers",
            "update_reservation_baggages",
            "cancel_pending_order",
            "modify_pending_order_address",
            "modify_pending_order_items",
            "modify_pending_order_payment",
            "modify_user_address",
            "return_delivered_order_items",
            "exchange_delivered_order_items",
            "send_certificate",
        }
        if tool_name not in write_tools:
            return True
        last_user = self._last_user_message(conversation)
        return self._is_affirmative(last_user)

    def _last_user_message(self, conversation: List[Dict[str, Any]]) -> str:
        for msg in reversed(conversation):
            if msg.get("role") == "user":
                return msg.get("content", "")
        return ""

    def _is_affirmative(self, text: str) -> bool:
        return bool(re.search(r"\b(yes|confirm|go ahead|proceed)\b", text, re.I))

    def _extract_known_identifiers(
        self,
        conversation: List[Dict[str, Any]],
        tool_outputs: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, str]:
        text = " ".join(msg.get("content", "") for msg in conversation)
        known: Dict[str, str] = {}
        # Flexible patterns for user id: "user id is X", "my user id is X", "user_id: X", etc.
        user_id = re.search(
            r"(?:my\s+)?user[\s_-]*id[:\s]+(?:is\s+)?([\w\-]+)", text, re.I
        )
        if user_id:
            known["user_id"] = user_id.group(1)
        # Flexible patterns for reservation id
        reservation_id = re.search(
            r"(?:my\s+)?reservation[\s_-]*(?:id|number)[:\s]+(?:is\s+)?([\w\-]+)", text, re.I
        )
        if reservation_id:
            known["reservation_id"] = reservation_id.group(1)
        # Flexible patterns for order id
        order_id = re.search(
            r"(?:my\s+)?order[\s_-]*(?:id|number)[:\s]+(?:is\s+)?([\w\-]+)", text, re.I
        )
        if order_id:
            known["order_id"] = order_id.group(1)
        # Email pattern
        email = re.search(
            r"[A-Z0-9._%+\-]+@[A-Z0-9.\-]+\.[A-Z]{2,}", text, re.I
        )
        if email:
            known["email"] = email.group(0)
        # Also extract identifiers from tool outputs (e.g. get_user_details response)
        if tool_outputs:
            for output in tool_outputs:
                output_text = str(output.get("output", ""))
                try:
                    output_data = json.loads(output_text) if isinstance(output_text, str) else output_text
                except (json.JSONDecodeError, TypeError):
                    output_data = {}
                if isinstance(output_data, dict):
                    if "user_id" in output_data and not known.get("user_id"):
                        known["user_id"] = str(output_data["user_id"])
                    if "email" in output_data and not known.get("email"):
                        known["email"] = str(output_data["email"])
                    for key in ("reservation_ids", "reservations"):
                        if key in output_data and not known.get("reservation_id"):
                            ids = output_data[key]
                            if isinstance(ids, list) and ids:
                                known["reservation_id"] = str(ids[0])
                    for key in ("order_ids", "orders"):
                        if key in output_data and not known.get("order_id"):
                            ids = output_data[key]
                            if isinstance(ids, list) and ids:
                                known["order_id"] = str(ids[0])
        return known
