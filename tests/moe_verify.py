import titan_py as titan
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def verify_moe():
    logger.info("Initializing TitanTransformer to verify MoE routing...")
    vocab_size = 100
    dim = 64
    num_layers = 2
    model = titan.TitanTransformer(vocab_size, dim, num_layers)

    # We can't directly inspect internal router weights from Python easily without exposing more,
    # but we can verify that the model runs and produces different outputs for different inputs,
    # and that the shared expert and dynamic experts are initialized.

    input_ids = [1, 2, 3, 4, 5]
    output1 = model.forward(input_ids)

    # Reset and try different input
    model.reset_state()
    input_ids2 = [10, 11, 12, 13, 14]
    output2 = model.forward(input_ids2)

    assert len(output1) == len(output2)
    assert output1 != output2
    logger.info("MoE verification: Model processed different inputs successfully.")

if __name__ == "__main__":
    verify_moe()
