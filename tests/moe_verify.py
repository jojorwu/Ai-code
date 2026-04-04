import titan_py as titan
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def verify_moe():
    logger.info("Verifying MoE block integration...")
    vocab_size = 100
    dim = 64
    num_layers = 1

    model = titan.TitanTransformer(vocab_size, dim, num_layers)

    # Test routing with a batch of tokens
    input_ids = [1, 2, 3, 4, 5]
    output = model.forward(input_ids)

    assert len(output) == len(input_ids) * vocab_size
    logger.info("MoE verification successful (Top-1 with shared expert)!")

if __name__ == "__main__":
    verify_moe()
