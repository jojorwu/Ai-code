import titan_py as titan

def train_demo():
    print("Titan Training Demo (Structural)")
    vocab_size = 1000
    dim = 128
    num_layers = 4

    model = titan.TitanTransformer(vocab_size, dim, num_layers)

    # Simulating a sequence of code snippets
    code_snippets = [
        "x = 10",
        "y = x + 5",
        "print(y)"
    ]

    print("Processing snippets with persistent state...")
    for i, snippet in enumerate(code_snippets):
        # In a real setup, we'd tokenize here
        # For demo, using dummy token IDs
        token_ids = [ord(c) % vocab_size for c in snippet]

        output = model.forward(token_ids)
        print(f"Snippet {i+1} processed. Output magnitude: {sum(abs(x) for x in output[:10]):.4f}")

    print("Training demo complete (Structural).")

if __name__ == "__main__":
    train_demo()
