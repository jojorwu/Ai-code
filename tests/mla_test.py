import titan_py as titan
import logging

logging.basicConfig(level=logging.INFO)

def test_mla():
    logging.info("Initializing TitanTransformer with Multi-Head Latent Attention (MLA)...")

    vocab_size = 1000
    dim = 256
    num_layers = 2

    # Initialize model with MLA
    model = titan.TitanTransformer(
        vocab_size,
        dim,
        num_layers,
        use_mla=True,
        kv_lora_rank=64,
        qk_lora_rank=32
    )

    logging.info("Running forward pass with MLA...")
    input_ids = [1, 2, 3, 4, 5]
    output = model.forward(input_ids)

    expected_size = 5 * vocab_size
    logging.info(f"Output size: {len(output)}")

    assert len(output) == expected_size, f"Expected size {expected_size}, got {len(output)}"
    logging.info("Success: Forward pass with MLA completed!")

if __name__ == "__main__":
    test_mla()
