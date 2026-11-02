# utils.py


def build_solver_prompt(instruction, question):
    return f"""You are solving a grade-school math word problem.

Instruction:
{instruction}

Question:
{question}

Requirements:
- Solve the problem carefully.
- The final line must be exactly in this format:
Final Answer: <number>
- Do not write anything after the final answer line.

Answer:
"""