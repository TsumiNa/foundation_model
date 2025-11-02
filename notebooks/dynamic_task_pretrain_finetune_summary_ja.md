# 実験概要（Dynamic Task Pretrain & PI Finetune Suite）

- **目的**: 非 PI 物性 15 種を段階的に追加しながら事前学習し、同一エンコーダを凍結したまま PI 物性 9 種を微調整することで、タスク順序と表現移転の影響を解析した。
- **データ**:
  - 記述子: `data/amorphous_polymer_FFDescriptor_20250730.parquet`
  - 非 PI 物性: `data/amorphous_polymer_non_PI_properties_20250730.parquet`
  - PI 物性: `data/amorphous_polymer_PI_properties_20250730.parquet`
  - 事前学習タスクは 15 物性（density 〜 thermal_diffusivity）をランダム順で使用。
  - 微調整タスクは density, Rg, r2, self-diffusion, Cp, Cv, linear_expansion, refractive_index, tg。
- **実験設計**:
  - 事前学習ラン数: 10。各ランで `RANDOM_SEED_BASE + run_idx` を用いてタスク順序とデータ分割を固定。
  - データ分割・マスキング: `CompoundDataModule` に統一シード (`DATAMODULE_RANDOM_SEED=42`) を渡し、任意の追加マスク (`TASK_MASKING_RATIOS`) と学習/検証スワップ (`SWAP_TRAIN_VAL_SPLIT`) に対応。
  - モデル: `FlexibleMultiTaskModel`。事前学習ではタスクを逐次追加し、微調整では共有エンコーダを凍結したまま各タスク専用ヘッドを追加。
  - 学習設定: 事前学習/微調整とも最大 200 エポック、`OptimizerConfig(lr=5e-2)`、早期終了と最良チェックポイント保存を適用。
- **評価**:
  - 各ステージ毎にテスト MAE を算出し、予測値・実測値のプロットとともに `artifacts/polymers_pretrain_finetune_runs/` 以下へ保存。
  - 微調整結果の `metrics.json` には、対応する事前学習ステージのタスク順序を記録し、タスク追加の影響を追跡可能にした。
- **補足**:
  - 追加解析ノート (`notebooks/finetune_mae_trajectory_analysis.ipynb`) で、各 PI 物性の MAE 推移とタスク追加時点を可視化。
  - ノートブック全体は `notebooks/dynamic_task_pretrain_finetune_suite.ipynb` に実装され、実験構成・再現手順をセル単位で管理している。

