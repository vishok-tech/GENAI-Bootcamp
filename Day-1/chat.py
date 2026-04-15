import ollama
import time

MODEL = "llama3.2:3b"

roles = {
    "1": {
        "name": "Python Tutor",
        "prompt": (
            "You are a patient and encouraging Python tutor. You explain concepts "
            "clearly with simple examples. If someone asks something unrelated to "
            "Python or programming, gently redirect them back to coding topics — "
            "but feel free to make a light joke or tie it back to Python if you can."
        ),
    },
    "2": {
        "name": "Fitness Coach",
        "prompt": (
            "You are an energetic and motivating fitness coach. You give practical, "
            "science-backed advice on exercise, posture, and healthy habits. Keep "
            "responses concise and actionable. Use an upbeat tone."
        ),
    },
    "3": {
        "name": "Travel Guide",
        "prompt": (
            "You are an enthusiastic world traveler and travel guide. You give vivid, "
            "insider tips about destinations, food, culture, and itineraries. Always "
            "inspire curiosity and wanderlust in your responses."
        ),
    },
    "4": {
        "name": "Stoic Philosopher",
        "prompt": (
            "You are a calm, wise Stoic philosopher in the tradition of Marcus Aurelius "
            "and Epictetus. You respond to all questions with thoughtful philosophical "
            "reflection, drawing on Stoic principles like focusing on what we control, "
            "accepting impermanence, and living virtuously."
        ),
    },
    "5": {
        "name": "Sarcastic Chef",
        "prompt": (
            "You are a brilliant but outrageously sarcastic chef. You have strong "
            "opinions about food and cooking. You answer all culinary questions with "
            "expert knowledge, but can't help being dramatically sarcastic about "
            "bad cooking choices. Think Gordon Ramsay, but funnier."
        ),
    },
}


def print_divider():
    print("\n" + "─" * 50 + "\n")


def show_roles():
    print("Available Roles:")
    for key, role in roles.items():
        print(f"  {key}. {role['name']}")
    # Show option to add custom role
    print(f"  c. ✏️  Create a custom role")


def pick_role():
    while True:
        show_roles()
        choice = input("\nPick a role (number or 'c'): ").strip().lower()

        if choice == "c":
            return create_custom_role()
        elif choice in roles:
            return choice
        else:
            print("❌ Invalid choice. Please try again.")


def create_custom_role():
    print("\n── Create a Custom Role ──")
    name = input("Role name (e.g. 'Pirate Navigator'): ").strip()
    if not name:
        name = "Custom Role"
    prompt = input("System prompt (describe how this role should behave):\n> ").strip()
    if not prompt:
        prompt = f"You are a helpful assistant called {name}."

    # Add to roles dict with a new key
    new_key = str(max(int(k) for k in roles if k.isdigit()) + 1)
    roles[new_key] = {"name": name, "prompt": prompt}
    print(f"\n✅ Custom role '{name}' added as option {new_key}!")
    return new_key


def chat_loop(role_key):
    role = roles[role_key]
    history = []  # conversation history (without the system message)

    print_divider()
    print(f"🎭 Role set: {role['name']}")
    print("Type your message, 'switch' to change role, 'roles' to add a custom role, or 'quit' to exit.")
    print_divider()

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\n👋 Goodbye!")
            return "quit"

        if not user_input:
            continue

        if user_input.lower() == "quit":
            print("\n👋 Goodbye!")
            return "quit"

        if user_input.lower() == "switch":
            print()
            return "switch"

        if user_input.lower() == "roles":
            print()
            new_key = create_custom_role()
            print(f"\nStill chatting as: {role['name']}")
            print("(Type 'switch' to switch to your new role)\n")
            continue

        # Append user message to history
        history.append({"role": "user", "content": user_input})

        # Build full messages list: system prompt + conversation history
        messages = [{"role": "system", "content": role["prompt"]}] + history

        # Call the model and time it
        start = time.time()
        try:
            response = ollama.chat(model=MODEL, messages=messages)
        except Exception as e:
            print(f"\n❌ Error contacting model: {e}")
            print("Make sure Ollama is running and the model is available.\n")
            history.pop()  # remove the user message we just added
            continue

        elapsed = time.time() - start

        assistant_reply = response["message"]["content"]

        # Append assistant reply to history
        history.append({"role": "assistant", "content": assistant_reply})

        # Display response with role name and timing
        token_count = response.get("eval_count", "?")
        print(f"\n{role['name']}: {assistant_reply}")
        print(f"\n  ⏱  {elapsed:.1f}s · {token_count} tokens\n")


def main():
    print("╔══════════════════════════════════╗")
    print("║       🤖  Role-Based Chat        ║")
    print("╚══════════════════════════════════╝\n")
    print(f"Model: {MODEL}\n")

    while True:
        role_key = pick_role()
        result = chat_loop(role_key)
        if result == "quit":
            break
        # result == "switch" → loop back to pick_role


if __name__ == "__main__":
    main()
