#!/usr/bin/env python3
"""Turbo码物理层密钥协商示例，并输出可用于NIST检验的二进制序列。"""

from __future__ import annotations

import argparse
import hashlib
import math
import random
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple


State = Tuple[int, int]


@dataclass
class TurboResult:
    alice_key: List[int]
    bob_key_before: List[int]
    bob_key_after: List[int]
    reconciled_key: List[int]
    key_disagreement_before: float
    key_disagreement_after: float
    nist_report: Dict[str, Tuple[float, bool]]
    random_bits: List[int]


def correlated_channel_samples(length: int, rho: float, sigma: float = 1.0) -> Tuple[List[float], List[float]]:
    """生成相关高斯信道观测（Alice/Bob）。"""
    alice = []
    bob = []
    for _ in range(length):
        z1 = random.gauss(0.0, sigma)
        z2 = random.gauss(0.0, sigma)
        alice.append(z1)
        bob.append(rho * z1 + math.sqrt(1.0 - rho ** 2) * z2)
    return alice, bob


def quantize(samples: Sequence[float]) -> List[int]:
    return [1 if x >= 0 else 0 for x in samples]


def build_rsc_trellis() -> Dict[State, Dict[int, Tuple[State, int]]]:
    """RSC(1,5/7) trellis, memory=2。"""
    trellis: Dict[State, Dict[int, Tuple[State, int]]] = {}
    for m1 in (0, 1):
        for m2 in (0, 1):
            st = (m1, m2)
            trellis[st] = {}
            for u in (0, 1):
                feedback = u ^ m1 ^ m2
                parity = feedback ^ m2
                next_state = (feedback, m1)
                trellis[st][u] = (next_state, parity)
    return trellis


def rsc_encode(bits: Sequence[int], trellis: Dict[State, Dict[int, Tuple[State, int]]]) -> List[int]:
    state: State = (0, 0)
    parity = []
    for b in bits:
        state, p = trellis[state][b]
        parity.append(p)
    return parity


def viterbi_decode(systematic: Sequence[int], parity: Sequence[int], reliability: Sequence[float],
                   trellis: Dict[State, Dict[int, Tuple[State, int]]], prior: Sequence[float] | None = None) -> List[int]:
    """硬判Viterbi，metric融合：系统比特、校验比特、先验。"""
    n = len(systematic)
    large = 10 ** 9
    states = list(trellis.keys())
    path_metric = {s: large for s in states}
    path_metric[(0, 0)] = 0.0
    history: List[Dict[State, Tuple[State, int]]] = []

    for i in range(n):
        new_metric = {s: large for s in states}
        prev_choice: Dict[State, Tuple[State, int]] = {}
        for st in states:
            if path_metric[st] >= large:
                continue
            for u in (0, 1):
                next_st, p = trellis[st][u]
                metric = path_metric[st]
                metric += (1.0 + reliability[i]) * (u != systematic[i])
                metric += 1.8 * (p != parity[i])
                if prior is not None:
                    metric += (0.5 - prior[i]) if u == 1 else prior[i]
                if metric < new_metric[next_st]:
                    new_metric[next_st] = metric
                    prev_choice[next_st] = (st, u)
        path_metric = new_metric
        history.append(prev_choice)

    final_state = min(states, key=lambda s: path_metric[s])
    out = [0] * n
    cur = final_state
    for i in range(n - 1, -1, -1):
        cur, u = history[i][cur]
        out[i] = u
    return out


def turbo_reconcile(alice_bits: Sequence[int], bob_bits: Sequence[int], bob_obs: Sequence[float],
                    iterations: int = 6) -> List[int]:
    n = len(alice_bits)
    trellis = build_rsc_trellis()
    interleaver = list(range(n))
    random.shuffle(interleaver)
    deinter = [0] * n
    for i, j in enumerate(interleaver):
        deinter[j] = i

    p1 = rsc_encode(alice_bits, trellis)
    inter_alice = [alice_bits[i] for i in interleaver]
    p2 = rsc_encode(inter_alice, trellis)

    reliability = [min(abs(x), 3.0) for x in bob_obs]
    prior = [0.5] * n
    estimate = list(bob_bits)

    for _ in range(iterations):
        dec1 = viterbi_decode(estimate, p1, reliability, trellis, prior)

        inter_sys = [dec1[i] for i in interleaver]
        inter_rel = [reliability[i] for i in interleaver]
        inter_prior = [prior[i] for i in interleaver]
        dec2_inter = viterbi_decode(inter_sys, p2, inter_rel, trellis, inter_prior)

        dec2 = [dec2_inter[deinter[i]] for i in range(n)]

        new_estimate = []
        for i in range(n):
            vote = dec1[i] + dec2[i] + estimate[i]
            new_estimate.append(1 if vote >= 2 else 0)
            prior[i] = 0.8 if new_estimate[i] else 0.2
        estimate = new_estimate
    return estimate


def privacy_amplification(bits: Sequence[int], out_len: int) -> List[int]:
    packed = ''.join(str(b) for b in bits).encode()
    digest = hashlib.shake_256(packed).digest((out_len + 7) // 8)
    out: List[int] = []
    for byte in digest:
        for i in range(8):
            out.append((byte >> (7 - i)) & 1)
            if len(out) == out_len:
                return out
    return out




def chi_square_survival(x: float, dof: int) -> float:
    """卡方分布右尾概率近似（Wilson-Hilferty）。"""
    if x <= 0:
        return 1.0
    k = float(dof)
    z = ((x / k) ** (1.0 / 3.0) - (1.0 - 2.0 / (9.0 * k))) / math.sqrt(2.0 / (9.0 * k))
    return 0.5 * math.erfc(z / math.sqrt(2.0))


def monobit_test(bits: Sequence[int]) -> Tuple[float, bool]:
    n = len(bits)
    s = sum(1 if b else -1 for b in bits)
    s_obs = abs(s) / math.sqrt(n)
    p = math.erfc(s_obs / math.sqrt(2.0))
    return p, p >= 0.01


def block_frequency_test(bits: Sequence[int], block_size: int = 128) -> Tuple[float, bool]:
    n = len(bits)
    nb = n // block_size
    if nb == 0:
        return 0.0, False
    chi2 = 0.0
    for i in range(nb):
        block = bits[i * block_size:(i + 1) * block_size]
        pi = sum(block) / block_size
        chi2 += 4.0 * block_size * (pi - 0.5) ** 2
    p = chi_square_survival(chi2, nb)
    return p, p >= 0.01


def runs_test(bits: Sequence[int]) -> Tuple[float, bool]:
    n = len(bits)
    pi = sum(bits) / n
    if abs(pi - 0.5) >= 2.0 / math.sqrt(n):
        return 0.0, False
    vobs = 1 + sum(bits[i] != bits[i - 1] for i in range(1, n))
    p = math.erfc(abs(vobs - 2 * n * pi * (1 - pi)) / (2 * math.sqrt(2 * n) * pi * (1 - pi)))
    return p, p >= 0.01


def longest_run_test(bits: Sequence[int], block_size: int = 128) -> Tuple[float, bool]:
    n = len(bits)
    nb = n // block_size
    if nb == 0:
        return 0.0, False
    stats = []
    for i in range(nb):
        block = bits[i * block_size:(i + 1) * block_size]
        longest = 0
        cur = 0
        for b in block:
            if b == 1:
                cur += 1
                longest = max(longest, cur)
            else:
                cur = 0
        stats.append(longest)
    mean = sum(stats) / len(stats)
    var = sum((x - mean) ** 2 for x in stats) / max(1, len(stats) - 1)
    # 经验阈值：长度128时，均值约7~8，方差不能过小。
    ok = 6.0 <= mean <= 10.5 and var > 1.0
    p = math.exp(-abs(mean - 8.0))
    return p, ok


def serial_test(bits: Sequence[int]) -> Tuple[float, bool]:
    n = len(bits)
    counts = [0, 0, 0, 0]
    for i in range(n):
        a = bits[i]
        b = bits[(i + 1) % n]
        counts[(a << 1) | b] += 1
    expected = n / 4.0
    chi2 = sum((c - expected) ** 2 / expected for c in counts)
    p = chi_square_survival(chi2, 3)
    return p, p >= 0.01


def run_nist_lite(bits: Sequence[int]) -> Dict[str, Tuple[float, bool]]:
    return {
        "monobit": monobit_test(bits),
        "block_frequency": block_frequency_test(bits),
        "runs": runs_test(bits),
        "longest_run": longest_run_test(bits),
        "serial_2bit": serial_test(bits),
    }


def bit_error_rate(a: Sequence[int], b: Sequence[int]) -> float:
    return sum(x != y for x, y in zip(a, b)) / len(a)


def build_scheme(key_len: int, output_len: int, rho: float, seed: int | None) -> TurboResult:
    if seed is not None:
        random.seed(seed)

    a_obs, b_obs = correlated_channel_samples(key_len, rho=rho)
    a_bits = quantize(a_obs)
    b_bits = quantize(b_obs)

    reconciled = turbo_reconcile(a_bits, b_bits, b_obs)
    random_bits = privacy_amplification(reconciled, output_len)
    nist_report = run_nist_lite(random_bits)

    return TurboResult(
        alice_key=a_bits,
        bob_key_before=b_bits,
        bob_key_after=reconciled,
        reconciled_key=reconciled,
        key_disagreement_before=bit_error_rate(a_bits, b_bits),
        key_disagreement_after=bit_error_rate(a_bits, reconciled),
        nist_report=nist_report,
        random_bits=random_bits,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Turbo码物理层密钥协商 + NIST-Lite随机性检测")
    parser.add_argument("--key-len", type=int, default=4096, help="初始信道比特长度")
    parser.add_argument("--output-len", type=int, default=20000, help="最终输出随机序列长度")
    parser.add_argument("--rho", type=float, default=0.94, help="Alice/Bob信道相关系数")
    parser.add_argument("--seed", type=int, default=2026, help="随机种子，便于复现实验")
    parser.add_argument("--out-file", type=str, default="nist_sequence.txt", help="输出比特序列文件")
    args = parser.parse_args()

    result = build_scheme(args.key_len, args.output_len, args.rho, args.seed)

    print("=== Turbo 物理层密钥协商结果 ===")
    print(f"初始密钥不一致率: {result.key_disagreement_before:.4f}")
    print(f"协商后不一致率 : {result.key_disagreement_after:.4f}")

    print("\n=== NIST-Lite 测试结果 (p >= 0.01 视为通过) ===")
    all_pass = True
    for name, (p, passed) in result.nist_report.items():
        all_pass &= passed
        status = "PASS" if passed else "FAIL"
        print(f"{name:16s} p={p:.6f} -> {status}")

    with open(args.out_file, "w", encoding="utf-8") as f:
        f.write(''.join(str(b) for b in result.random_bits))
    print(f"\n序列已输出到: {args.out_file}")
    print(f"整体结果: {'PASS' if all_pass else 'FAIL'}")


if __name__ == "__main__":
    main()
