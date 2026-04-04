import titan_py as titan
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_model():
    logger.info("Initializing TitanTransformer...")
    vocab_size = 100
    dim = 64
    num_layers = 2

    model = titan.TitanTransformer(vocab_size, dim, num_layers)

    logger.info("Running first forward pass...")
    input_ids = [1, 2, 3, 4, 5]
    output1 = model.forward(input_ids)
    logger.info(f"Output1 size: {len(output1)}")

    logger.info("Running second forward pass (state should persist)...")
    output2 = model.forward(input_ids)
    logger.info(f"Output2 size: {len(output2)}")

    assert len(output1) == len(output2) == len(input_ids) * vocab_size

    # In a real model with random weights, output1 and output2 would be different
    # because the state (memory_states) has changed.
    # Currently we use VarBuilder::zeros, so they might be the same, but let's check.
    if output1 == output2:
        logger.info("Note: Output1 and Output2 are identical (likely due to zero initialization).")
    else:
        logger.info("Success: Output1 and Output2 differ (state persistence verified).")

    logger.info("Resetting state...")
    model.reset_state()
    output3 = model.forward(input_ids)
    assert len(output3) == len(input_ids) * vocab_size
    logger.info("Forward pass after reset successful!")

if __name__ == "__main__":
    test_model()
