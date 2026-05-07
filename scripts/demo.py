"""End-to-end demo reproducing the case study from Section 4.

Run with:
    python scripts/demo.py
"""

import logging

from membrane.model import simulator, workload

logger = logging.getLogger(__name__)


def main():
    logger.info("=" * 60)
    logger.info("Membrane-PD Case Study Reproduction (arXiv:2604.15039v2)")
    logger.info("=" * 60)

    # Generate workload: truncated log-normal, mean ~27K tokens
    logger.info("\n[1] Generating workload ...")
    lengths = workload.generate_request_lengths(20000, seed=42)
    mean_len = sum(lengths) / len(lengths)
    logger.info(f"    Sample size: {len(lengths)}")
    logger.info(f"    Mean input length: {mean_len:.0f} tokens")

    # Run Membrane-PD
    logger.info("\n[2] Running Membrane-PD simulation ...")
    result = simulator.run_membrane_pd(lengths)
    logger.info(f"    Threshold t          : {result.threshold:,} tokens")
    logger.info(
        f"    N_membrane / N_p / N_d : {result.num_membrane} / {result.num_pd_p} / {result.num_pd_d}"
    )
    logger.info(f"    Theta_membrane         : {result.theta_membrane:.2f} req/s")
    logger.info(f"    Theta_pd-p           : {result.theta_pd_p:.2f} req/s")
    logger.info(f"    Theta_pd-d           : {result.theta_pd_d:.2f} req/s")
    logger.info(f"    Lambda_max           : {result.lambda_max:.2f} req/s")
    logger.info(f"    Mean TTFT            : {result.mean_ttft:.2f} s")
    logger.info(f"    P90 TTFT             : {result.p90_ttft:.2f} s")
    logger.info(f"    Avg egress bandwidth : {result.bandwidth_gbps:.1f} Gbps")
    logger.info(f"    Fraction to Membrane   : {result.fraction_to_membrane:.1%}")

    # Run homogeneous PD baseline
    logger.info("\n[3] Running Homogeneous PD baseline ...")
    hom = simulator.run_homogeneous_pd(lengths)
    logger.info(f"    N_p / N_d            : {hom.num_pd_p} / {hom.num_pd_d}")
    logger.info(f"    Theta_pd-p           : {hom.theta_pd_p:.2f} req/s")
    logger.info(f"    Theta_pd-d           : {hom.theta_pd_d:.2f} req/s")
    logger.info(f"    Lambda_max           : {hom.lambda_max:.2f} req/s")
    logger.info(f"    Mean TTFT            : {hom.mean_ttft:.2f} s")
    logger.info(f"    P90 TTFT             : {hom.p90_ttft:.2f} s")

    # Run naive heterogeneous PD baseline
    logger.info("\n[4] Running Naive Heterogeneous PD baseline ...")
    naive = simulator.run_naive_heterogeneous_pd(lengths)
    logger.info(f"    N_membrane / N_d       : {naive.num_membrane} / {naive.num_pd_d}")
    logger.info(f"    Theta_membrane         : {naive.theta_membrane:.2f} req/s")
    logger.info(f"    Theta_pd-d           : {naive.theta_pd_d:.2f} req/s")
    logger.info(f"    Lambda_max           : {naive.lambda_max:.2f} req/s")
    logger.info(f"    Mean TTFT            : {naive.mean_ttft:.2f} s")
    logger.info(f"    P90 TTFT             : {naive.p90_ttft:.2f} s")
    logger.info(f"    Avg egress bandwidth : {naive.bandwidth_gbps:.1f} Gbps")

    # Comparison
    logger.info("\n[5] Comparison vs. Homogeneous PD baseline")
    logger.info(f"    Throughput gain      : {result.lambda_max / hom.lambda_max:.2f}x")
    logger.info(
        f"    Mean TTFT reduction  : {(1 - result.mean_ttft / hom.mean_ttft) * 100:.0f}%"
    )
    logger.info(
        f"    P90 TTFT reduction   : {(1 - result.p90_ttft / hom.p90_ttft) * 100:.0f}%"
    )

    logger.info("\n" + "=" * 60)
    logger.info("Demo complete.")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
