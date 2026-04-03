import titan_py as titan
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_inference():
    logger.info("Initializing TitanTransformer for inference...")
    vocab_size = 100
    dim = 64
    num_layers = 2

    model = titan.TitanTransformer(vocab_size, dim, num_layers)

    # Initial prompt
    input_ids = [1, 2, 3]
    logger.info(f"Input: {input_ids}")

    # First token generation
    output1 = model.forward(input_ids)
    # Get last token logits
    last_logits = output1[-vocab_size:]
    next_token = last_logits.index(max(last_logits))
    logger.info(f"Generated token 1: {next_token}")

    # Second token generation (incremental using KV-cache)
    # The model.forward call handles the cache internally in the current implementation
    output2 = model.forward([next_token])
    last_logits2 = output2[-vocab_size:]
    next_token2 = last_logits2.index(max(last_logits2))
    logger.info(f"Generated token 2: {next_token2}")

    # Verify that we can reset and start over
    model.reset_state()
    output3 = model.forward(input_ids)
    assert len(output3) == len(output1)
    logger.info("Inference test successful!")

if __name__ == "__main__":
    test_inference()
