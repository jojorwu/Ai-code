import titan_py as titan
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def benchmark_context():
    logger.info("Starting Context Efficiency Benchmark...")
    vocab_size = 100
    dim = 64
    num_layers = 2

    model = titan.TitanTransformer(vocab_size, dim, num_layers)

    # Test with a very long sequence
    seq_len = 3000
    input_ids = [i % vocab_size for i in range(seq_len)]

    start_time = time.time()
    output = model.forward(input_ids)
    end_time = time.time()

    logger.info(f"Forward pass for {seq_len} tokens took {end_time - start_time:.4f} seconds.")
    assert len(output) == seq_len * vocab_size

    logger.info("Context efficiency test successful!")

if __name__ == "__main__":
    benchmark_context()
