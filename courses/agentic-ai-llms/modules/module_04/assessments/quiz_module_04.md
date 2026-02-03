# Quiz: Module 4 - Planning & Reasoning

**Estimated Time:** 18 minutes
**Total Points:** 100
**Passing Score:** 70%

---

## Instructions

This quiz evaluates your understanding of the ReAct pattern, goal decomposition, and self-reflection in agent systems. Focus on demonstrating practical application of these reasoning patterns.

---

## Part A: ReAct Pattern (35 points)

### Question 1 (10 points)
Describe the ReAct (Reasoning + Acting) loop and explain how it differs from simple question-answering.

**Your Answer:**

---

### Question 2 (12 points)
Given the following goal, write out a ReAct trace showing Thought → Action → Observation for at least 2 iterations:

**Goal:** "Find out how many employees work at the company that created the ChatGPT API"

**Your Answer:**

---

### Question 3 (8 points)
What is the purpose of the "Thought" step in ReAct, and why not skip straight to "Action"?

**Your Answer:**

---

### Question 4 (5 points)
**True or False:** In a ReAct agent, the LLM executes the actions directly during the reasoning process.

**Your Answer:**

---

## Part B: Goal Decomposition (35 points)

### Question 5 (10 points)
What is goal decomposition, and why is it essential for complex multi-step tasks?

**Your Answer:**

---

### Question 6 (12 points)
Decompose the following complex goal into 4-6 concrete subtasks with clear dependencies:

**Goal:** "Research competitors' pricing strategies and create a pricing recommendation report"

**Your Answer:**

---

### Question 7 (8 points)
Compare "Plan-and-Execute" vs "Adaptive Planning" approaches. When would you use each?

**Your Answer:**

---

### Question 8 (5 points)
Which planning approach is best for a dynamic environment where new information frequently changes the optimal path forward?

A) Plan-and-Execute (create full plan upfront)
B) ReAct (interleave thinking and acting)
C) Hierarchical planning (nested sub-plans)
D) Random exploration

**Your Answer:**

---

## Part C: Self-Reflection (30 points)

### Question 9 (10 points)
Describe the self-reflection loop for agents. What are the key steps, and what problem does it solve?

**Your Answer:**

---

### Question 10 (12 points)
Given this agent execution scenario, write a self-reflection prompt that would help the agent identify and fix the issue:

**Scenario:**
```
User: "Book me a flight to Paris for next Friday"
Agent Action: search_flights(destination="Paris", date="2024-02-09")
Result: Error - No flights found
Agent Response: "Sorry, no flights available to Paris"
```

The agent failed to specify departure city and gave up after one error. Write a reflection prompt.

**Your Answer:**

---

### Question 11 (8 points)
What are the trade-offs of adding self-reflection to an agent system?

Benefits:
Drawbacks:

**Your Answer:**

---

## Part D: Planning Strategies (Bonus - 10 points)

### Question 12 (5 points)
What is hierarchical planning, and when is it more effective than flat task decomposition?

**Your Answer:**

---

### Question 13 (5 points)
You're building an agent that writes code, tests it, and fixes bugs. This might require multiple iterations. Which pattern is most appropriate?

A) Simple ReAct with no planning
B) Plan-and-Execute with fixed steps
C) ReAct with self-reflection and adaptive replanning
D) Random trial and error

**Your Answer:**

---

# Answer Key

## Part A: ReAct Pattern

**Question 1:** (10 points)
*Sample Answer:* ReAct is a reasoning pattern where agents alternate between Thinking (reasoning about what to do), Acting (executing tools/actions), and Observing (processing results). The loop continues until reaching a final answer. Unlike simple Q&A where the model directly answers from training data, ReAct agents actively gather information, reason about it, and make decisions based on real-time observations. This enables solving problems requiring external information or multi-step reasoning.

*Grading:*
- Explains loop structure (3 pts)
- Describes how it differs from Q&A (4 pts)
- Mentions information gathering (3 pts)

**Question 2:** (12 points)
*Sample Answer:*
```
Goal: Find out how many employees work at the company that created the ChatGPT API

Thought 1: First, I need to identify which company created the ChatGPT API.
Action 1: search("who created ChatGPT API")
Observation 1: OpenAI created ChatGPT and provides the ChatGPT API.

Thought 2: Now I need to find how many employees work at OpenAI.
Action 2: search("OpenAI number of employees")
Observation 2: OpenAI has approximately 700-800 employees as of 2024.

Thought 3: I have the information needed to answer the question.
Final Answer: OpenAI, the company that created the ChatGPT API, has approximately 700-800 employees.
```

*Grading:*
- Two complete iterations (4 pts each = 8 pts)
- Clear thought process (2 pts)
- Proper conclusion (2 pts)

**Question 3:** (8 points)
*Sample Answer:* The "Thought" step provides explicit reasoning about what to do next and why. It serves several purposes: (1) Makes agent behavior interpretable and debuggable, (2) Allows the model to consider options before committing to action, (3) Provides context for subsequent steps, and (4) Enables the model to recognize when it has sufficient information to conclude. Skipping to Action would make the agent reactive rather than deliberative.

**Question 4:** False (5 points)
*Explanation:* The LLM generates text describing actions (structured as tool calls), but your code executes the actual actions. The model only predicts what action should be taken—it doesn't execute anything itself.

## Part B: Goal Decomposition

**Question 5:** (10 points)
*Sample Answer:* Goal decomposition is breaking complex goals into smaller, manageable subtasks with clear objectives. It's essential because: (1) LLMs are better at solving smaller, focused problems than massive complex ones, (2) Allows progress tracking and partial success, (3) Enables parallel execution of independent subtasks, (4) Makes debugging easier by isolating failures to specific steps, and (5) Reduces cognitive load on the model.

**Question 6:** (12 points)
*Sample Answer:*
1. **Identify competitors** - Research and list main competitors in market
2. **Gather competitor pricing data** - For each competitor, collect pricing information (depends on step 1)
3. **Analyze pricing patterns** - Identify pricing strategies, tiers, and positioning (depends on step 2)
4. **Assess our product positioning** - Determine where our product fits in the market
5. **Develop pricing recommendation** - Based on analysis, create data-driven recommendation (depends on steps 3-4)
6. **Create report** - Structure findings into professional report (depends on steps 1-5)

*Grading:*
- 4-6 logical subtasks (6 pts)
- Clear dependencies noted (3 pts)
- Specific and actionable (3 pts)

**Question 7:** (8 points)
*Sample Answer:*
**Plan-and-Execute:** Creates complete plan upfront, then executes steps sequentially. Use when: (1) Requirements are well-defined and stable, (2) Environment is predictable, (3) Steps are clearly ordered.

**Adaptive Planning:** Creates initial plan but revises based on observations and outcomes. Use when: (1) Environment is dynamic, (2) Intermediate results affect next steps, (3) Uncertainty exists about optimal approach.

**Question 8:** Answer: B (5 points)
*Explanation:* ReAct's interleaved thinking and acting allows the agent to adapt based on observations at each step, making it ideal for dynamic environments where plans may need to change.

## Part C: Self-Reflection

**Question 9:** (10 points)
*Sample Answer:* The self-reflection loop consists of:
1. **Execute Plan** - Agent attempts task
2. **Evaluate Outcome** - Assess whether goal was achieved
3. **Identify Issues** - Analyze what went wrong (if anything)
4. **Revise Approach** - Modify strategy based on learnings
5. **Retry** - Attempt again with improved approach

This solves the problem of agents making mistakes and giving up. Instead of failing immediately, the agent can learn from failures, recognize errors, and iteratively improve its approach.

**Question 10:** (12 points)
*Sample Answer:*
```
Reflect on your previous attempt to book a flight. Consider:

1. Did you gather all necessary information from the user before searching?
   - What information was missing from the query?

2. When you encountered an error, did you diagnose the root cause?
   - Why might the search have failed?
   - What assumptions did you make?

3. What alternative actions could you have taken?
   - What clarifying questions could you ask?
   - How could you make the search more likely to succeed?

4. Based on this reflection, what should you do differently?
   - Formulate an improved plan to accomplish the user's goal.
```

*Grading:*
- Identifies missing information (departure city) (3 pts)
- Prompts consideration of alternatives (3 pts)
- Encourages diagnosis of error (3 pts)
- Asks for revised approach (3 pts)

**Question 11:** (8 points)
*Sample Answer:*

**Benefits:**
- Improved accuracy through error correction
- Better handling of edge cases and failures
- More robust agent behavior
- Ability to learn from mistakes within a session

**Drawbacks:**
- Increased latency (additional LLM calls)
- Higher token costs (reflection adds overhead)
- Risk of infinite loops if reflection is poor
- More complex implementation and debugging

*Grading:* 2 points per benefit/drawback (minimum 2 of each)

## Part D: Planning Strategies

**Question 12:** (5 points)
*Sample Answer:* Hierarchical planning creates nested plans with high-level goals broken into sub-goals, which are further decomposed. It's more effective than flat decomposition for very complex projects where subtasks themselves require multi-step execution. For example, "Build a web app" → "Design database" → ["Create schema", "Set up migrations", "Seed data"].

**Question 13:** Answer: C (5 points)
*Explanation:* Code generation with testing requires iteration and adaptation based on test results. ReAct provides the action loop, while self-reflection enables learning from test failures to improve the code. Adaptive replanning allows the agent to revise its approach based on what works.

---

## Scoring Guide

- **90-100 points:** Excellent - Strong grasp of planning and reasoning patterns
- **80-89 points:** Good - Solid understanding with room for refinement
- **70-79 points:** Passing - Core concepts understood, practice implementation
- **Below 70:** Review module and implement ReAct agent

**Key Topics to Review if Struggling:**
- ReAct loop structure (Thought → Action → Observation)
- Goal decomposition strategies
- Difference between plan-and-execute vs adaptive planning
- Self-reflection loop and when to use it
- Hierarchical vs flat task decomposition
- Trade-offs of different planning approaches
