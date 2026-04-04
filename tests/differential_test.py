import titan_py as titan
import logging

logging.basicConfig(level=logging.INFO)

def test_differential_attention():
    logging.info("Initializing TitanTransformer with Differential Attention...")

    # Large vocab and dim for testing
    vocab_size = 1000
    dim = 256
    num_layers = 2

    # Initialize model using positional and keyword arguments as per the wrapper
    model = titan.TitanTransformer(
        vocab_size,
        dim,
        num_layers,
        use_differential_attn=True
    )

    logging.info("Running forward pass with Differential Attention...")
    input_ids = [1, 2, 3, 4, 5]
    output = model.forward(input_ids)

    # model.forward returns a flattened list of logits
    # Output size should be len(input_ids) * vocab_size
    expected_size = 5 * vocab_size
    logging.info(f"Output size: {len(output)}")

    assert len(output) == expected_size, f"Expected size {expected_size}, got {len(output)}"
    logging.info("Success: Forward pass with Differential Attention completed!")

if __name__ == "__main__":
    test_differential_attention()
