# Triage Summary — Trả lời / Phản biện / Code thêm

Quick reference. Chi tiết xem `Revision_Roadmap.md` và `Response_Letter_Skeleton.md`.

| ID | Issue | Posture | Effort |
|---|---|---|---|
| **R1-W1** | Novelty vs MoRE | **PHẢN BIỆN** (positioning) | 1 paragraph in §2.3 |
| **R1-W2** | Scope hẹp (backbone, multi-task, retrieval) | **CODE + PHẢN BIỆN** | ViT-L/14 runs + Future Work paragraph |
| **R1-W3** | Multi-head chỉ chạy được H=2 | **TRẢ LỜI** (mechanistic) | Add 2 paragraphs in §4.3.1 |
| **R1-W4** | Thiếu baseline MoRE/O-LoRA/DoRA | **CODE (DoRA, SingLoRA) + PHẢN BIỆN (MoRE/O-LoRA scope mismatch)** | Run DoRA + SingLoRA |
| **R1-Q1** | Tại sao H>2 fail dưới r=2/4 + ortho strong | **TRẢ LỜI** (rank fragmentation + gate variance) | Same as W3 |
| **R1-Q2** | Multi-task future work | **TRẢ LỜI** (deferred) | Expand §5 |
| **R1-Q3** | ViT-L/14 scaling | **CODE** | Combine with W2 |
| **R2-M1** | PSD claim sai (softmax giữ PSD) | **THỪA NHẬN + VIẾT LẠI** (optional: signed-gating ablation) | Major rewrite Abstract/Intro/§3.2.2/Fig.2 |
| **R2-M2** | Hyperparam chọn trên test set | **CODE (re-do trên val) + THỪA NHẬN** | Re-run H, λ_o sweeps on val |
| **R2-M3** | Seed protocol mâu thuẫn + gain trong noise | **CODE (3 seeds + paired test) + THỪA NHẬN** | **Lớn nhất**: re-run all main tables |
| **R2-M4** | Fig 7 sai rank budget (r=2 mà có 15 singular values) | **CODE (regenerate) + THỪA NHẬN** | Regenerate spectral fig với config đúng |
| **R2-m5a** | Numerical inconsistencies (3.6%, EuroSAT 88.64/89.06/88.57, Table 4 dupe, dấu phẩy thập phân) | **VIẾT** (fix) | Cross-check logs |
| **R2-m5b** | SingLoRA full baseline + sharper OMoE | **CODE + VIẾT** | Trùng với R1-W4 |
| **R2-m5c** | u(t), ε không định nghĩa | **VIẾT** | 1-line fixes |
| **R2-m5d** | Thiếu ImageNet/SUN397/StanfordCars | **CODE** | Run 3 dataset bổ sung |
| **R2-m5e** | Release code + seeds | **THỪA NHẬN** | Update Data Availability statement |

## Tổng kết theo posture

**THỪA NHẬN HOÀN TOÀN (concede):** R2-M1 (PSD), R2-M2 (val), R2-M3 (seeds), R2-M4 (Fig 7), R2-m5a, R2-m5c, R2-m5e
→ Đây là các issue đúng về kỹ thuật, không cãi được. Cần viết lại tử tế.

**TRẢ LỜI (answer with existing evidence):** R1-W3/Q1 (mechanistic), R1-Q2 (future work)
→ Paper đã có đủ data, chỉ cần viết rõ hơn.

**PHẢN BIỆN (pushback with justification):** R1-W1 (novelty positioning), R1-W2 (multi-task/retrieval scope), R1-W4 part (MoRE/O-LoRA không fair to reproduce)
→ Giữ vững quan điểm nhưng phải concede những phần valid.

**CẦN CHẠY CODE THÊM (must run):**
1. **3 seeds toàn bộ Tables 1-3** — non-negotiable (R2-M3) — ~24 GPU-h
2. **ViT-L/14** trên 3 dataset @ 4-shot (R1-W2/Q3) — ~16 GPU-h
3. **ImageNet + SUN397 + StanfordCars** @ 4-shot (R2-m5d) — ~12 GPU-h (ImageNet expensive)
4. **DoRA + SingLoRA baselines** đầy đủ (R1-W4, R2-m5b) — ~16 GPU-h
5. **Val-split sweep cho H và λ_o** (R2-M2) — ~6 GPU-h
6. **Regenerate Fig 7** với r=2, H=2 checkpoint chính xác (R2-M4) — minutes
7. **(Optional)** Signed-gating ablation (R2-M1) — ~4 GPU-h — recommended if time permits

**Total compute estimate: ~80 A100-hours**, doable trong 3-4 tuần với deadline 2026-07-15.
