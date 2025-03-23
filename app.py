import streamlit as st
from openai import OpenAI

###############################################################################
# 1) SETUP
###############################################################################

st.set_page_config(
    page_title="AI-Powered Influence Simulation",
    layout="centered",
)

# Create the OpenAI client
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
# Ensure these session states exist
if "scenario_output" not in st.session_state:
    st.session_state["scenario_output"] = None
if "evaluation_rubric" not in st.session_state:
    st.session_state["evaluation_rubric"] = None
if "simulation_messages" not in st.session_state:
    st.session_state["simulation_messages"] = []
if "evaluation_feedback" not in st.session_state:
    st.session_state["evaluation_feedback"] = None
if "simulation_finished" not in st.session_state:
    st.session_state["simulation_finished"] = False

###############################################################################
# 2) INFLUENCE DATA
###############################################################################
INFLUENCE_DATA = """
[ ... Your Influence Data goes here in full ... ]
"""

###############################################################################
# 3) PROMPT TEMPLATES
###############################################################################

SCENARIO_DESIGNER_PROMPT_TEMPLATE = """
You are the Scenario Designer Agent (o3-mini). 
You have 'reasoning_effort="high"' to produce a thoughtful scenario.

User inputs:
- Influence tactics to practice: {tactics}
- User scenario details: {scenario_details}
- Desired role: {role}
- Make it up? {make_it_up}
- Difficulty: {difficulty}

Use the following Influence Data to inform your scenario design:
{influence_data}

Tasks:
1. Create a realistic, detailed scenario requiring the user to apply at least two different influence tactics.
2. Produce a multi-criteria evaluation rubric (scoring or rating) that will be used to judge the user's performance.
3. Output must contain:
   - SCENARIO:
   - EVALUATION_RUBRIC:
"""

SIMULATION_FACILITATOR_PROMPT_TEMPLATE = """
You are the Simulation Facilitator Agent (gpt-4o).
Scenario:
{scenario_text}

Incorporate relevant influence frameworks, relationships, power tactics, etc.
Conversation so far:
{conversation_history}

Respond with the next scenario development and ask the user, “What will you do next?” 
Invite them to respond in paragraphs.
"""

EVALUATOR_PROMPT_TEMPLATE = """
You are the Evaluator & Feedback Agent (gpt-4.5).

Scenario:
{scenario_text}

User-Facilitator Conversation:
{conversation_history}

Evaluation Rubric:
{evaluation_rubric}

1) Score the user’s performance against the rubric.
2) Provide detailed feedback (strengths, weaknesses).
3) Reference relevant influence frameworks in your explanation.
"""

###############################################################################
# 4) AGENT-CALLING FUNCTIONS
###############################################################################

def call_scenario_designer(tactics, scenario_details, role, make_it_up, difficulty):
    """Calls the Scenario Designer agent. Returns scenario + rubric text."""
    prompt = SCENARIO_DESIGNER_PROMPT_TEMPLATE.format(
        tactics=tactics,
        scenario_details=scenario_details,
        role=role,
        make_it_up=make_it_up,
        difficulty=difficulty,
        influence_data=INFLUENCE_DATA
    )
    response = client.chat.completions.create(
        model="o3-mini",
        reasoning_effort="high",
        messages=[
            {"role": "system", "content": "You are a helpful scenario designer."},
            {"role": "user", "content": prompt},
        ]
    )
    text_out = response.choices[0].message.content

    # Parse SCENARIO: and EVALUATION_RUBRIC:
    scenario_part, rubric_part = "", ""
    if "EVALUATION_RUBRIC:" in text_out:
        parts = text_out.split("EVALUATION_RUBRIC:")
        scenario_part = parts[0].replace("SCENARIO:", "").strip()
        rubric_part = parts[1].strip()
    else:
        scenario_part = text_out.strip()
        rubric_part = "No rubric found."
    return scenario_part, rubric_part


def call_simulation_facilitator(scenario_text, conversation_text):
    """Calls the Simulation Facilitator (gpt-4o) to get next scenario step."""
    prompt = SIMULATION_FACILITATOR_PROMPT_TEMPLATE.format(
        scenario_text=scenario_text,
        conversation_history=conversation_text
    )
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are the simulation facilitator."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.7
    )
    return response.choices[0].message.content


def call_evaluator(scenario_text, conversation_text, rubric_text):
    """Calls the Evaluator & Feedback agent (gpt-4.5)."""
    prompt = EVALUATOR_PROMPT_TEMPLATE.format(
        scenario_text=scenario_text,
        conversation_history=conversation_text,
        evaluation_rubric=rubric_text
    )
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are the evaluator."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.7
    )
    return response.choices[0].message.content

###############################################################################
# 5) MAIN STREAMLIT APP
###############################################################################

def main():
    st.title("AI-Powered Influence Simulation")

    ###########################################################################
    # (A) SCENARIO SETUP -- only shown if no scenario is yet generated
    ###########################################################################
    if st.session_state["scenario_output"] is None:
        st.header("Step 1: Scenario Setup")
        user_tactics = st.text_input("Which influence tactics do you want to practice?")
        user_scenario = st.text_area("Describe an existing scenario you'd like to retry (or leave blank).")
        user_role = st.text_input("What role do you want to play?")
        user_make_up = st.radio("Should the AI just make something up?", ["Yes", "No"])
        user_difficulty = st.slider("How challenging do you want it?", 1, 5, 3)

        if st.button("Generate Scenario"):
            scenario_text, rubric_text = call_scenario_designer(
                tactics=user_tactics,
                scenario_details=user_scenario,
                role=user_role,
                make_it_up=user_make_up,
                difficulty=user_difficulty
            )
            st.session_state["scenario_output"] = scenario_text
            st.session_state["evaluation_rubric"] = rubric_text
            st.success("Scenario Generated! Please scroll down to continue.")

    ###########################################################################
    # (B) SHOW SCENARIO + SIMULATION UI (Chat Bot) if we have scenario
    ###########################################################################
    if st.session_state["scenario_output"] is not None and not st.session_state["simulation_finished"]:
        st.header("Step 2: Scenario Simulation")
        scenario_text = st.session_state["scenario_output"]
        st.markdown("#### Scenario Context:")
        st.write(scenario_text)

        st.markdown("---")
        st.markdown("### Simulation Conversation")

        # Show conversation so far
        for msg in st.session_state["simulation_messages"]:
            if msg["role"] == "user":
                st.markdown(f"**You:** {msg['content']}")
            else:
                st.markdown(f"**Simulation Facilitator:** {msg['content']}")

        st.markdown("---")
        user_input = st.text_area("Your next move (paragraph or more):")

        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("Submit Move"):
                if not user_input.strip():
                    st.warning("Please enter a move.")
                else:
                    # Append user's message
                    st.session_state["simulation_messages"].append(
                        {"role": "user", "content": user_input}
                    )
                    # Build conversation text
                    conversation_text = ""
                    for m in st.session_state["simulation_messages"]:
                        role_label = "USER" if m["role"] == "user" else "FACILITATOR"
                        conversation_text += f"{role_label}: {m['content']}\n"

                    # Call the facilitator
                    facilitator_output = call_simulation_facilitator(scenario_text, conversation_text)
                    st.session_state["simulation_messages"].append(
                        {"role": "assistant", "content": facilitator_output}
                    )
                    st.success("Move submitted! Scroll down to see the facilitator's response.")

        with col2:
            if st.button("Finish & Evaluate"):
                st.session_state["simulation_finished"] = True
                st.success("You've chosen to finish. Scroll to Step 3 for evaluation.")

    ###########################################################################
    # (C) EVALUATION UI if user finishes simulation
    ###########################################################################
    if st.session_state["simulation_finished"]:
        st.header("Step 3: Evaluation & Feedback")
        # If we haven't evaluated yet, do so
        if st.session_state["evaluation_feedback"] is None:
            st.info("Evaluating your entire conversation... please wait.")

            # Build conversation text
            conversation_text = ""
            for m in st.session_state["simulation_messages"]:
                role_label = "USER" if m["role"] == "user" else "FACILITATOR"
                conversation_text += f"{role_label}: {m['content']}\n"

            # Call evaluator
            feedback = call_evaluator(
                st.session_state["scenario_output"],
                conversation_text,
                st.session_state["evaluation_rubric"]
            )
            st.session_state["evaluation_feedback"] = feedback
            st.success("Evaluation complete! See below for detailed feedback.")

        # Show feedback
        if st.session_state["evaluation_feedback"]:
            st.markdown("### Evaluation Results")
            st.write(st.session_state["evaluation_feedback"])

        # Restart button
        if st.button("Restart Everything"):
            st.session_state["scenario_output"] = None
            st.session_state["evaluation_rubric"] = None
            st.session_state["simulation_messages"] = []
            st.session_state["evaluation_feedback"] = None
            st.session_state["simulation_finished"] = False
            st.success("All cleared. Scroll up to Step 1 to begin again.")

###############################################################################
# 6) RUN
###############################################################################
if __name__ == "__main__":
    main()
