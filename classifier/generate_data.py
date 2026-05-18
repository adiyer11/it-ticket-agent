"""
classifier/generate_data.py
Uses an LLM to generate realistic synthetic IT support tickets per category.
This mirrors the prompt-engineered training data generation built at Cisco.
"""

import argparse
import csv
import os
from openai import OpenAI

client = OpenAI()  # reads OPENAI_API_KEY from env

SYSTEM_PROMPT = """You are an IT support ticket generator. 
Generate realistic, diverse IT support ticket texts that an employee might submit.
Vary the tone (frustrated, polite, brief, detailed), phrasing, and technical specificity.
Return ONLY a JSON array of strings — no other text."""

def generate_tickets(category: str, n: int) -> list[str]:
    prompt = (
        f"Generate {n} realistic IT support ticket texts for the category: '{category}'.\n"
        f"Tickets should sound like real employee messages — varied in length and tone.\n"
        f"Return a JSON array of {n} strings."
    )
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0.9,
        response_format={"type": "json_object"},
    )
    import json
    content = response.choices[0].message.content
    data = json.loads(content)
    # handle both {"tickets": [...]} and bare list
    if isinstance(data, list):
        return data
    return next(iter(data.values()))


def main(categories: list[str], n_per_category: int, output_path: str):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    rows = []
    for cat in categories:
        print(f"Generating {n_per_category} tickets for: {cat}")
        tickets = generate_tickets(cat, n_per_category)
        for text in tickets:
            rows.append({"text": text.strip(), "category": cat})

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["text", "category"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved {len(rows)} tickets to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--categories",
        default="password_reset,access_request,hardware,software_install,onboarding",
        help="Comma-separated list of ticket categories",
    )
    parser.add_argument("--n", type=int, default=200,
                        help="Tickets to generate per category")
    parser.add_argument("--output", default="data/tickets.csv")
    args = parser.parse_args()

    cats = [c.strip() for c in args.categories.split(",")]
    main(cats, args.n, args.output)
