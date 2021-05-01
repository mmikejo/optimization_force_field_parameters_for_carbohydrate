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

# 入力データ (Google Drive https://drive.google.com/drive/folders/1jAbYIfcm9tVxtbwxN_IMbM1gdB1LOR6T?usp=sharing )
- 0GB.charge.txt <br>
 GLYCAM06 での &beta;-D-glucose の電荷ファイル。パラメータ最適化前後の比較に使用。
- 0GB.remove_duplication.1_4_eel.dat <br>
 1-4 静電相互作用に関連した距離情報とその原子間に働く静電相互作用エネルギー。<br>
 コードの実際の入力データである delta_energy_profile_edft_opt_reduce_datasets.bf.csv の作成に使用。
- 0GB.remove_duplication.edft.dat <br>
 B3LYP/6-31++G(2d,2p) 精度でのQM計算によるエネルギー。<br>
 コードの実際の入力データである delta_energy_profile_edft_opt_reduce_datasets.bf.csv の作成に使用。
- 0GB.remove_duplication.eel.dat <br>
 静電相互作用に関連した距離情報とその原子間に働く静電相互作用エネルギー。<br>
 コードの実際の入力データである delta_energy_profile_edft_opt_reduce_datasets.bf.csv の作成に使用。
- 0GB.remove_duplication.ene.mm_all.dat <br>
 MM のbond,angle,torsion,vdw14,elec14,vdw,elecごとのエネルギーと総エネルギー。<br>
 コードの実際の入力データである delta_energy_profile_edft_opt_reduce_datasets.bf.csv の作成に使用。
- 0GB.remove_duplication.tor.dat <br>
 Torsion項に関連した二面角情報とその原子間に働くTorsionエネルギー。<br>
 コードの実際の入力データである delta_energy_profile_edft_opt_reduce_datasets.bf.csv の作成に使用。
- charge_distribution.csv <br>
 入力に用いた31899構造の RESP 電荷の平均と標準偏差。パラメータ最適化のペナルティー項として使用。
- delta_energy_profile_edft_opt_reduce_datasets.bf.csv <br>
 コードの入力データ。<br>


# 必要要件
以下の実行環境およびライブラリでの動作を確認 <br>
Python : 3.6.6 <br>
tensorflow : 1.11.0 <br>
pandas : 0.23.4 <br>
matplotlib : 3.0.0 <br>
numpy : 1.14.5 <br>
scikit-learn : 0.20.0 <br>
