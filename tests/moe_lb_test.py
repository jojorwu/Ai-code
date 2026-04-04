import titan_py as titan
import logging

logging.basicConfig(level=logging.INFO)

def test_moe_lb():
    logging.info("Initializing TitanTransformer with Auxiliary-loss-free Load Balancing...")

    vocab_size = 1000
    dim = 256
    num_layers = 2
    num_experts = 8

    # Initialize model with Aux-loss-free LB
    model = titan.TitanTransformer(
        vocab_size,
        dim,
        num_layers,
        num_experts=num_experts,
        use_aux_loss_free_lb=True
    )

    logging.info("Running forward pass with MoE LB...")
    input_ids = [1, 2, 3, 4, 5]
    output = model.forward(input_ids)

    expected_size = 5 * vocab_size
    logging.info(f"Output size: {len(output)}")

    assert len(output) == expected_size, f"Expected size {expected_size}, got {len(output)}"
    logging.info("Success: Forward pass with MoE LB completed!")

if __name__ == "__main__":
    test_moe_lb()
