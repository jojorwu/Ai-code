import titan_py as titan
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_load_stability():
    vocab_size = 100
    dim = 64
    num_layers = 2
    model = titan.TitanTransformer(vocab_size, dim, num_layers)

    # 1. Normal Case
    logger.info("Running normal sequence...")
    model.forward([1, 2, 3])

    # 2. Border Case (Max Seq Len)
    logger.info("Running max allowed sequence...")
    max_seq = [1] * 8192
    model.forward(max_seq)

    # 3. Violation Case (Exceeding Max Seq Len)
    logger.info("Verifying error for too long sequence...")
    too_long = [1] * 8193
    try:
        model.forward(too_long)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        logger.info(f"Caught expected error: {e}")

    # 4. Training Constraint
    logger.info("Verifying training length constraint...")
    too_long_train = [1] * 4097
    try:
        model.train_step(too_long_train, too_long_train)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        logger.info(f"Caught expected error: {e}")

    logger.info("Load stability test passed!")

if __name__ == "__main__":
    test_load_stability()
