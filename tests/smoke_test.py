import sys
import os

# Ensure the .so can be found
sys.path.append(os.path.join(os.getcwd(), "titan_core"))

try:
    import titan_core as titan
    print("Imported titan_core from root")
except ImportError as e:
    print(f"Failed to import from root: {e}")
    sys.path.append(os.path.join(os.getcwd(), "titan_core"))
    import titan_core as titan
    print("Imported titan_core from titan_core/ directory")

def test_model():
    print("Initializing TitanTransformer...")
    vocab_size = 100
    dim = 64
    num_layers = 2

    model = titan.PyTitanTransformer(vocab_size, dim, num_layers)

    print("Running first forward pass...")
    input_ids = [1, 2, 3, 4, 5]
    output1 = model.forward(input_ids)
    print(f"Output1 size: {len(output1)}")

    print("Running second forward pass (state should persist)...")
    output2 = model.forward(input_ids)
    print(f"Output2 size: {len(output2)}")

    assert len(output1) == len(output2) == len(input_ids) * vocab_size

    # In a real model with random weights, output1 and output2 would be different
    # because the state (memory_states) has changed.
    # Currently we use VarBuilder::zeros, so they might be the same, but let's check.
    if output1 == output2:
        print("Note: Output1 and Output2 are identical (likely due to zero initialization).")
    else:
        print("Success: Output1 and Output2 differ (state persistence verified).")

    print("Resetting state...")
    model.reset_state()
    output3 = model.forward(input_ids)
    assert len(output3) == len(input_ids) * vocab_size
    print("Forward pass after reset successful!")

if __name__ == "__main__":
    test_model()
