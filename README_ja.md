# optimization_force_field_parameters_for_carbohydrate

# 構成
- README_ja.md <br>
- 0GB.nowat.prmtop <br>
 &beta;-D-glucose のAMBER 用のパラメータトポロジーファイル
- 0GB.remoe_duplication.mdcrd <br>
 パラメータ最適化に用いた構造の座標データ、上記のトポロジーファイルに対応
- beta_D_glucose_lowest_energy_confomer_with_b3lyp.pdb <br>
全サンプル中DFT計算での電子エネルギーが最安定だった構造のPDBファイル
- charge_torsion_fit_template.ipynb <br>
パラメータフィッティングに用いたコードを Jupyter のノートブック形式、コメントアウトは現在日本語
- charge_torsion_fit_tor6period.py <br>
上記のノートブックから作成した Python 用のコード
