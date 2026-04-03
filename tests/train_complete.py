import titan_py as titan

def train_complete_demo():
    print("Titan Complete Training Demo")
    vocab_size = 100
    dim = 64
    num_layers = 2

    model = titan.TitanTransformer(vocab_size, dim, num_layers)
    model.init_optimizer(lr=1e-3)

    # Dummy dataset: predict next character
    # "hello" -> inputs: "hell", targets: "ello"
    text = "hello"
    x_ids = [ord(c) % vocab_size for c in text[:-1]]
    target_ids = [ord(c) % vocab_size for c in text[1:]]

    print(f"Training on: '{text}'")
    for epoch in range(10):
        model.reset_state()
        loss = model.train_step(x_ids, target_ids)
        print(f"Epoch {epoch+1}, Loss: {loss:.4f}")

    weights_path = "model_weights.safetensors"
    print(f"Saving weights to {weights_path}...")
    model.save_weights(weights_path)

    print("Loading weights into a new model...")
    new_model = titan.TitanTransformer(vocab_size, dim, num_layers)
    new_model.load_weights(weights_path)

    print("Training demo complete.")

if __name__ == "__main__":
    train_complete_demo()
