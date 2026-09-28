# RISA P0 formal validation analysis

Checkpoints were selected by validation accuracy. Val CE is post-hoc only.
Post-hoc Macro-F1 uses class IDs observed in train/validation labels only; test labels are not read.
Stored checkpoint Macro-F1 remains recorded separately as the original selection-time metric.
Test and LP evaluation were disabled.

## Movies / seed 42 / p0_identity

- normal: Acc=0.57138568, Macro-F1=0.49076635, CE=1.38465655

## Movies / seed 43 / p0_identity

- normal: Acc=0.57888418, Macro-F1=0.49245065, CE=1.37712634

## Movies / seed 44 / p0_identity

- normal: Acc=0.57318532, Macro-F1=0.46510677, CE=1.34068453

## Movies / seed 42 / p0_masspres_scalar

- normal: Acc=0.58188361, Macro-F1=0.48371744, CE=1.39636791

## Movies / seed 43 / p0_masspres_scalar

- normal: Acc=0.57318532, Macro-F1=0.47315745, CE=1.39622760

## Movies / seed 44 / p0_masspres_scalar

- normal: Acc=0.57828432, Macro-F1=0.49858493, CE=1.41042387

## Movies / seed 42 / p0_operator_uniform

- normal: Acc=0.57318532, Macro-F1=0.48431865, CE=1.36383963

## Movies / seed 43 / p0_operator_uniform

- normal: Acc=0.57918411, Macro-F1=0.48328346, CE=1.46290851

## Movies / seed 44 / p0_operator_uniform

- normal: Acc=0.58068383, Macro-F1=0.47686776, CE=1.35328734

## Movies / seed 42 / p0_operator_global

- normal: Acc=0.57558489, Macro-F1=0.48903954, CE=1.35434353

## Movies / seed 43 / p0_operator_global

- normal: Acc=0.57678461, Macro-F1=0.49466146, CE=1.38645661

## Movies / seed 44 / p0_operator_global

- normal: Acc=0.58008397, Macro-F1=0.48677528, CE=1.36873567

## Movies / seed 42 / p0_operator_routed

- normal: Acc=0.57888418, Macro-F1=0.44865784, CE=1.33558881
- router_mean: Acc=0.57978404, Macro-F1=0.44989252, CE=1.33530843
- router_uniform: Acc=0.57468504, Macro-F1=0.44141725, CE=1.33771956
- router_matched_shuffle: Acc=0.57888418, Macro-F1=0.44846947, CE=1.33532810
- delta_off: Acc=0.50449908, Macro-F1=0.24433804, CE=1.82093012
- delta_parallel_only: Acc=0.50509894, Macro-F1=0.24313318, CE=1.73531580
- delta_orthogonal_only: Acc=0.57438511, Macro-F1=0.44037088, CE=1.36397755
- Routed diagnostics: `{"text": {"correction_norm_ratio": 0.4113226532936096, "cos_base_base_plus_delta": 0.9124256850039179, "mean_kl_pi_edge_to_mean_pi": 0.005141227132637205, "mean_router_entropy": 1.3138610026205009, "operator_output_norm": [2.9793968200683594, 5.902662754058838, 4.719487190246582, 2.684826612472534], "orthogonal_fraction": 0.8904414176940918, "orthogonal_norm_ratio": 0.36625874042510986, "pairwise_operator_output_cosine": [0.9828367233276367, 0.9884674549102783, 0.983654797077179, 0.9942149519920349, 0.9627634286880493, 0.974086344242096], "parallel_norm_ratio": 0.18719208240509033, "per_operator_across_edge_routing_std": [0.023083522679849228, 0.0415485813410778, 0.004342841771116224, 0.021097961988661876], "per_operator_mean_routing_probability": [0.15840987193925424, 0.3970098759864519, 0.2648049764927951, 0.17977527573574428]}, "visual": {"correction_norm_ratio": 0.28061360120773315, "cos_base_base_plus_delta": 0.9605277375281402, "mean_kl_pi_edge_to_mean_pi": 0.026696412682031934, "mean_router_entropy": 1.3061314458614495, "operator_output_norm": [2.821542739868164, 3.006838321685791, 4.242895603179932, 2.946831703186035], "orthogonal_fraction": 0.9192723035812378, "orthogonal_norm_ratio": 0.25796031951904297, "pairwise_operator_output_cosine": [0.8891783356666565, 0.9487178921699524, 0.90642249584198, 0.8854575157165527, 0.8469409346580505, 0.8975463509559631], "parallel_norm_ratio": 0.11045584082603455, "per_operator_across_edge_routing_std": [0.026628297057715907, 0.025409470895878927, 0.06317474453333989, 0.10339587997291982], "per_operator_mean_routing_probability": [0.1959753426215606, 0.20857287556827428, 0.19674937089480748, 0.398702410819099]}}`
- Matched shuffle: `{"edge_count": 160802, "method": "deterministic cyclic shift within target-degree/source-degree/normalized-weight quantile bins", "non_singleton_bin_count": 37, "preserves_edge_marginal_within_each_bin": true, "quantile_bins_per_feature": 4, "quantile_cutpoints": {"normalized_weight": [0.03042903169989586, 0.04767312854528427, 0.07905694097280502], "source_degree": [10.0, 19.0, 36.0], "target_degree": [10.0, 19.0, 36.0]}, "shuffled_edge_count": 160802, "singleton_edge_count": 0, "structural_bin_count": 37}`

## Movies / seed 43 / p0_operator_routed

- normal: Acc=0.57708454, Macro-F1=0.46832059, CE=1.35137689
- router_mean: Acc=0.57588482, Macro-F1=0.46690114, CE=1.35792851
- router_uniform: Acc=0.55848831, Macro-F1=0.42673328, CE=1.41348886
- router_matched_shuffle: Acc=0.57708454, Macro-F1=0.47121866, CE=1.35115504
- delta_off: Acc=0.50179964, Macro-F1=0.25908836, CE=1.93391097
- delta_parallel_only: Acc=0.50089979, Macro-F1=0.25740118, CE=1.82515478
- delta_orthogonal_only: Acc=0.57438511, Macro-F1=0.45935607, CE=1.39448559
- Routed diagnostics: `{"text": {"correction_norm_ratio": 0.45568716526031494, "cos_base_base_plus_delta": 0.8911079308715065, "mean_kl_pi_edge_to_mean_pi": 0.05395891489758531, "mean_router_entropy": 0.501933354391999, "operator_output_norm": [2.7848474979400635, 3.16367769241333, 2.515223741531372, 5.279786586761475], "orthogonal_fraction": 0.8706420660018921, "orthogonal_norm_ratio": 0.3967404067516327, "pairwise_operator_output_cosine": [0.9721704721450806, 0.9583552479743958, 0.943405270576477, 0.9439997673034668, 0.967698335647583, 0.89387047290802], "parallel_norm_ratio": 0.22416022419929504, "per_operator_across_edge_routing_std": [0.02920156211439167, 0.04576233271668573, 0.04795309632694114, 0.12155741144054627], "per_operator_mean_routing_probability": [0.01939165404575241, 0.059198692856879544, 0.06564848675838636, 0.8557611664160839]}, "visual": {"correction_norm_ratio": 0.2877027988433838, "cos_base_base_plus_delta": 0.9585862971231701, "mean_kl_pi_edge_to_mean_pi": 0.1389262821060553, "mean_router_entropy": 1.221876129457454, "operator_output_norm": [3.4416587352752686, 3.806098222732544, 2.728754997253418, 2.3953371047973633], "orthogonal_fraction": 0.9075601100921631, "orthogonal_norm_ratio": 0.26110759377479553, "pairwise_operator_output_cosine": [0.8786839842796326, 0.8677055239677429, 0.7439675331115723, 0.6891061067581177, 0.6198599338531494, 0.8823861479759216], "parallel_norm_ratio": 0.12081275135278702, "per_operator_across_edge_routing_std": [0.0873104584161919, 0.23664716780709205, 0.1100744194384434, 0.05257522503592627], "per_operator_mean_routing_probability": [0.2736719392324931, 0.32208296579410545, 0.23407068223766275, 0.17017441281574572]}}`
- Matched shuffle: `{"edge_count": 160802, "method": "deterministic cyclic shift within target-degree/source-degree/normalized-weight quantile bins", "non_singleton_bin_count": 37, "preserves_edge_marginal_within_each_bin": true, "quantile_bins_per_feature": 4, "quantile_cutpoints": {"normalized_weight": [0.03042903169989586, 0.04767312854528427, 0.07905694097280502], "source_degree": [10.0, 19.0, 36.0], "target_degree": [10.0, 19.0, 36.0]}, "shuffled_edge_count": 160802, "singleton_edge_count": 0, "structural_bin_count": 37}`

## Movies / seed 44 / p0_operator_routed

- normal: Acc=0.57618475, Macro-F1=0.47454092, CE=1.40176535
- router_mean: Acc=0.57468504, Macro-F1=0.46105899, CE=1.39142275
- router_uniform: Acc=0.56328732, Macro-F1=0.46068066, CE=1.40116739
- router_matched_shuffle: Acc=0.57558489, Macro-F1=0.46368104, CE=1.39308560
- delta_off: Acc=0.49340129, Macro-F1=0.24662128, CE=2.11585689
- delta_parallel_only: Acc=0.49610075, Macro-F1=0.25290602, CE=1.99417353
- delta_orthogonal_only: Acc=0.57468504, Macro-F1=0.46269297, CE=1.45546532
- Routed diagnostics: `{"text": {"correction_norm_ratio": 0.40033721923828125, "cos_base_base_plus_delta": 0.9175364928607853, "mean_kl_pi_edge_to_mean_pi": 0.0013922523873749829, "mean_router_entropy": 0.05129373926160281, "operator_output_norm": [3.3429160118103027, 4.226237773895264, 4.818561553955078, 4.965407371520996], "orthogonal_fraction": 0.8773950338363647, "orthogonal_norm_ratio": 0.3512538969516754, "pairwise_operator_output_cosine": [0.8523927927017212, 0.972050130367279, 0.9712481498718262, 0.8915587663650513, 0.9186860918998718, 0.9745728373527527], "parallel_norm_ratio": 0.19206923246383667, "per_operator_across_edge_routing_std": [0.00021356437990233926, 0.0051645669029539215, 0.0007461091052047835, 0.004229752272516375], "per_operator_mean_routing_probability": [0.0001733270914398884, 0.9914662208771792, 0.0008230740456878434, 0.007537377830482114]}, "visual": {"correction_norm_ratio": 0.3028595447540283, "cos_base_base_plus_delta": 0.9542074647703387, "mean_kl_pi_edge_to_mean_pi": 0.19551023288638025, "mean_router_entropy": 1.1741024643850744, "operator_output_norm": [3.9185690879821777, 2.842343807220459, 1.6413133144378662, 5.684221267700195], "orthogonal_fraction": 0.9076199531555176, "orthogonal_norm_ratio": 0.27488136291503906, "pairwise_operator_output_cosine": [0.9013822078704834, 0.36172032356262207, 0.9572661519050598, 0.4803447127342224, 0.8853447437286377, 0.25360384583473206], "parallel_norm_ratio": 0.12713833153247833, "per_operator_across_edge_routing_std": [0.07100030476280014, 0.06710209996806958, 0.2111103234274246, 0.22047952537085558], "per_operator_mean_routing_probability": [0.20863389159150447, 0.20594814310348053, 0.31694299563047995, 0.26847496981419455]}}`
- Matched shuffle: `{"edge_count": 160802, "method": "deterministic cyclic shift within target-degree/source-degree/normalized-weight quantile bins", "non_singleton_bin_count": 37, "preserves_edge_marginal_within_each_bin": true, "quantile_bins_per_feature": 4, "quantile_cutpoints": {"normalized_weight": [0.03042903169989586, 0.04767312854528427, 0.07905694097280502], "source_degree": [10.0, 19.0, 36.0], "target_degree": [10.0, 19.0, 36.0]}, "shuffled_edge_count": 160802, "singleton_edge_count": 0, "structural_bin_count": 37}`

## ele-fashion / seed 42 / p0_identity

- normal: Acc=0.87685382, Macro-F1=0.75595488, CE=0.42066765

## ele-fashion / seed 43 / p0_identity

- normal: Acc=0.87572879, Macro-F1=0.75182499, CE=0.41898003

## ele-fashion / seed 44 / p0_identity

- normal: Acc=0.87675160, Macro-F1=0.74863147, CE=0.43603617

## ele-fashion / seed 42 / p0_masspres_scalar

- normal: Acc=0.87562650, Macro-F1=0.75330608, CE=0.42309174

## ele-fashion / seed 43 / p0_masspres_scalar

- normal: Acc=0.87767208, Macro-F1=0.76473488, CE=0.44202337

## ele-fashion / seed 44 / p0_masspres_scalar

- normal: Acc=0.87746751, Macro-F1=0.75603774, CE=0.41808882

## ele-fashion / seed 42 / p0_operator_uniform

- normal: Acc=0.87859261, Macro-F1=0.74655850, CE=0.40838483

## ele-fashion / seed 43 / p0_operator_uniform

- normal: Acc=0.88114965, Macro-F1=0.76295346, CE=0.42583454

## ele-fashion / seed 44 / p0_operator_uniform

- normal: Acc=0.87716067, Macro-F1=0.74414204, CE=0.40682811

## ele-fashion / seed 42 / p0_operator_global

- normal: Acc=0.87828577, Macro-F1=0.75486234, CE=0.40350446

## ele-fashion / seed 43 / p0_operator_global

- normal: Acc=0.87736523, Macro-F1=0.75407731, CE=0.39333251

## ele-fashion / seed 44 / p0_operator_global

- normal: Acc=0.87767208, Macro-F1=0.75707327, CE=0.40217704

## ele-fashion / seed 42 / p0_operator_routed

- normal: Acc=0.87828577, Macro-F1=0.75021766, CE=0.45569021
- router_mean: Acc=0.87388772, Macro-F1=0.74693051, CE=0.45159090
- router_uniform: Acc=0.85343152, Macro-F1=0.73770195, CE=0.52048761
- router_matched_shuffle: Acc=0.87716067, Macro-F1=0.74894548, CE=0.45139346
- delta_off: Acc=0.86662579, Macro-F1=0.73728749, CE=0.55470926
- delta_parallel_only: Acc=0.86918277, Macro-F1=0.74505293, CE=0.50965428
- delta_orthogonal_only: Acc=0.87787664, Macro-F1=0.75040751, CE=0.49045995
- Routed diagnostics: `{"text": {"correction_norm_ratio": 0.46224549412727356, "cos_base_base_plus_delta": 0.8894525730762679, "mean_kl_pi_edge_to_mean_pi": 0.3415319586994882, "mean_router_entropy": 0.3323344833128257, "operator_output_norm": [6.9831862449646, 2.3446197509765625, 2.986508369445801, 5.1756591796875], "orthogonal_fraction": 0.8753405809402466, "orthogonal_norm_ratio": 0.40462225675582886, "pairwise_operator_output_cosine": [0.5380244255065918, 0.6597929000854492, 0.45431405305862427, 0.8744983077049255, 0.6354260444641113, 0.6529475450515747], "parallel_norm_ratio": 0.22349876165390015, "per_operator_across_edge_routing_std": [0.3482277834528593, 0.005713412356922346, 0.007640995510075724, 0.35428183570782507], "per_operator_mean_routing_probability": [0.30444149388404734, 0.004324972068362987, 0.00569518077086123, 0.6855383527641479]}, "visual": {"correction_norm_ratio": 0.4669964611530304, "cos_base_base_plus_delta": 0.8869876268876574, "mean_kl_pi_edge_to_mean_pi": 0.31167966354244625, "mean_router_entropy": 0.8333408552752263, "operator_output_norm": [5.759669780731201, 5.787477016448975, 5.0181427001953125, 6.116580486297607], "orthogonal_fraction": 0.8872281312942505, "orthogonal_norm_ratio": 0.41433241963386536, "pairwise_operator_output_cosine": [0.22310516238212585, 0.29971548914909363, 0.6290577054023743, 0.8649960160255432, 0.4993020296096802, 0.4426896870136261], "parallel_norm_ratio": 0.21543997526168823, "per_operator_across_edge_routing_std": [0.3253676569686059, 0.14528960405218408, 0.11873141752133408, 0.14138526109243724], "per_operator_mean_routing_probability": [0.5770596108923256, 0.14017095709044367, 0.15232143402770973, 0.13044799784948485]}}`
- Matched shuffle: `{"edge_count": 399172, "method": "deterministic cyclic shift within target-degree/source-degree/normalized-weight quantile bins", "non_singleton_bin_count": 35, "preserves_edge_marginal_within_each_bin": true, "quantile_bins_per_feature": 4, "quantile_cutpoints": {"normalized_weight": [0.044543538242578506, 0.07537783682346344, 0.13130642473697662], "source_degree": [3.0, 9.0, 34.0], "target_degree": [3.0, 9.0, 34.0]}, "shuffled_edge_count": 399172, "singleton_edge_count": 0, "structural_bin_count": 35}`

## ele-fashion / seed 43 / p0_operator_routed

- normal: Acc=0.87787664, Macro-F1=0.75632847, CE=0.39901817
- router_mean: Acc=0.87777436, Macro-F1=0.75384462, CE=0.39819425
- router_uniform: Acc=0.86386418, Macro-F1=0.74548141, CE=0.44276062
- router_matched_shuffle: Acc=0.87746751, Macro-F1=0.75493907, CE=0.39820987
- delta_off: Acc=0.85895473, Macro-F1=0.73594720, CE=0.49786136
- delta_parallel_only: Acc=0.86151171, Macro-F1=0.73880688, CE=0.46405742
- delta_orthogonal_only: Acc=0.87572879, Macro-F1=0.74927681, CE=0.42106435
- Routed diagnostics: `{"text": {"correction_norm_ratio": 0.44243523478507996, "cos_base_base_plus_delta": 0.899248240733318, "mean_kl_pi_edge_to_mean_pi": 0.11227516027760792, "mean_router_entropy": 0.23679102300989524, "operator_output_norm": [5.32600736618042, 1.6876999139785767, 3.3110127449035645, 6.201143264770508], "orthogonal_fraction": 0.8602104783058167, "orthogonal_norm_ratio": 0.3805874288082123, "pairwise_operator_output_cosine": [0.5451830625534058, 0.609329342842102, 0.45232027769088745, 0.711817741394043, 0.753055989742279, 0.7045356035232544], "parallel_norm_ratio": 0.22561505436897278, "per_operator_across_edge_routing_std": [0.15644635328034626, 0.01307930719584233, 0.022182944399737368, 0.12623565467498749], "per_operator_mean_routing_probability": [0.9082736551531337, 0.0032185930461362477, 0.008776588977193165, 0.07973116343102088]}, "visual": {"correction_norm_ratio": 0.5585940480232239, "cos_base_base_plus_delta": 0.8341440218752818, "mean_kl_pi_edge_to_mean_pi": 8.186967996308288e-11, "mean_router_entropy": 1.023383384702753e-12, "operator_output_norm": [1.9666844606399536, 3.2139315605163574, 6.432928085327148, 1.7441554069519043], "orthogonal_fraction": 0.8770010471343994, "orthogonal_norm_ratio": 0.4898875653743744, "pairwise_operator_output_cosine": [0.8093674182891846, 0.7683204412460327, 0.8916140794754028, 0.7766135334968567, 0.8891347050666809, 0.8196454048156738], "parallel_norm_ratio": 0.26839807629585266, "per_operator_across_edge_routing_std": [1.0012038523669693e-15, 2.3910730601491797e-13, 0.0, 4.218362932494538e-15], "per_operator_mean_routing_probability": [7.985291368256298e-17, 3.694657962137516e-14, 1.0, 3.9823819598903684e-16]}}`
- Matched shuffle: `{"edge_count": 399172, "method": "deterministic cyclic shift within target-degree/source-degree/normalized-weight quantile bins", "non_singleton_bin_count": 35, "preserves_edge_marginal_within_each_bin": true, "quantile_bins_per_feature": 4, "quantile_cutpoints": {"normalized_weight": [0.044543538242578506, 0.07537783682346344, 0.13130642473697662], "source_degree": [3.0, 9.0, 34.0], "target_degree": [3.0, 9.0, 34.0]}, "shuffled_edge_count": 399172, "singleton_edge_count": 0, "structural_bin_count": 35}`

## ele-fashion / seed 44 / p0_operator_routed

- normal: Acc=0.87664932, Macro-F1=0.75025711, CE=0.40787607
- router_mean: Acc=0.87685382, Macro-F1=0.75065059, CE=0.40720743
- router_uniform: Acc=0.87030786, Macro-F1=0.75108688, CE=0.43963966
- router_matched_shuffle: Acc=0.87675160, Macro-F1=0.75063268, CE=0.40752965
- delta_off: Acc=0.85956836, Macro-F1=0.72793619, CE=0.50133044
- delta_parallel_only: Acc=0.86273909, Macro-F1=0.73300119, CE=0.46986276
- delta_orthogonal_only: Acc=0.87593335, Macro-F1=0.74726910, CE=0.42810905
- Routed diagnostics: `{"text": {"correction_norm_ratio": 0.4342724680900574, "cos_base_base_plus_delta": 0.9023010444119327, "mean_kl_pi_edge_to_mean_pi": 1.798193749932619e-10, "mean_router_entropy": 1.5709919050938239e-09, "operator_output_norm": [2.2253942489624023, 5.166550159454346, 1.27956223487854, 2.3666887283325195], "orthogonal_fraction": 0.8992490768432617, "orthogonal_norm_ratio": 0.3905191123485565, "pairwise_operator_output_cosine": [0.5947164297103882, 0.6650567650794983, 0.5504586100578308, 0.47178196907043457, 0.6091328859329224, 0.3675345778465271], "parallel_norm_ratio": 0.18996687233448029, "per_operator_across_edge_routing_std": [2.5315687313444444e-10, 0.0, 6.191886946882481e-13, 4.5859602870421266e-11], "per_operator_mean_routing_probability": [6.22933668804724e-11, 1.0, 9.261473650166606e-14, 1.0248959406305133e-11]}, "visual": {"correction_norm_ratio": 0.4485388696193695, "cos_base_base_plus_delta": 0.8961799362179712, "mean_kl_pi_edge_to_mean_pi": 0.018919484246438772, "mean_router_entropy": 0.04032388857488935, "operator_output_norm": [2.264064073562622, 5.134907245635986, 4.928515434265137, 1.2515887022018433], "orthogonal_fraction": 0.8764885663986206, "orthogonal_norm_ratio": 0.39313918352127075, "pairwise_operator_output_cosine": [0.6357818245887756, 0.7107731103897095, 0.8587819933891296, 0.6474097967147827, 0.5753138065338135, 0.7831023335456848], "parallel_norm_ratio": 0.2159367799758911, "per_operator_across_edge_routing_std": [0.0024827446398989206, 0.03944284568657352, 0.0353954185950999, 0.0021999157943074538], "per_operator_mean_routing_probability": [0.0002559734331754846, 0.9897898763563636, 0.009727255756502063, 0.00022689483378612432]}}`
- Matched shuffle: `{"edge_count": 399172, "method": "deterministic cyclic shift within target-degree/source-degree/normalized-weight quantile bins", "non_singleton_bin_count": 35, "preserves_edge_marginal_within_each_bin": true, "quantile_bins_per_feature": 4, "quantile_cutpoints": {"normalized_weight": [0.044543538242578506, 0.07537783682346344, 0.13130642473697662], "source_degree": [3.0, 9.0, 34.0], "target_degree": [3.0, 9.0, 34.0]}, "shuffled_edge_count": 399172, "singleton_edge_count": 0, "structural_bin_count": 35}`

## Reddit-S / seed 42 / p0_identity

- normal: Acc=0.95942122, Macro-F1=0.91809064, CE=0.16566783

## Reddit-S / seed 43 / p0_identity

- normal: Acc=0.95942122, Macro-F1=0.92452790, CE=0.16243893

## Reddit-S / seed 44 / p0_identity

- normal: Acc=0.96005034, Macro-F1=0.92769436, CE=0.16515833

## Reddit-S / seed 42 / p0_masspres_scalar

- normal: Acc=0.96130860, Macro-F1=0.92967195, CE=0.16549046

## Reddit-S / seed 43 / p0_masspres_scalar

- normal: Acc=0.96005034, Macro-F1=0.92856859, CE=0.16808183

## Reddit-S / seed 44 / p0_masspres_scalar

- normal: Acc=0.96099401, Macro-F1=0.92571549, CE=0.19049735

## Reddit-S / seed 42 / p0_operator_uniform

- normal: Acc=0.95973575, Macro-F1=0.92723835, CE=0.18576397

## Reddit-S / seed 43 / p0_operator_uniform

- normal: Acc=0.96099401, Macro-F1=0.92965945, CE=0.16491126

## Reddit-S / seed 44 / p0_operator_uniform

- normal: Acc=0.96193773, Macro-F1=0.92877527, CE=0.16365841

## Reddit-S / seed 42 / p0_operator_global

- normal: Acc=0.95942122, Macro-F1=0.92051281, CE=0.16439903

## Reddit-S / seed 43 / p0_operator_global

- normal: Acc=0.96099401, Macro-F1=0.92432099, CE=0.16348948

## Reddit-S / seed 44 / p0_operator_global

- normal: Acc=0.96193773, Macro-F1=0.93072138, CE=0.17735618

## Reddit-S / seed 42 / p0_operator_routed

- normal: Acc=0.96036488, Macro-F1=0.92157313, CE=0.16122216
- router_mean: Acc=0.95784837, Macro-F1=0.91907457, CE=0.15923129
- router_uniform: Acc=0.95721924, Macro-F1=0.91592569, CE=0.16299543
- router_matched_shuffle: Acc=0.95942122, Macro-F1=0.91928287, CE=0.15627602
- delta_off: Acc=0.95501733, Macro-F1=0.91432701, CE=0.16715074
- delta_parallel_only: Acc=0.95690471, Macro-F1=0.91719837, CE=0.16717620
- delta_orthogonal_only: Acc=0.96036488, Macro-F1=0.92255022, CE=0.15904045
- Routed diagnostics: `{"text": {"correction_norm_ratio": 0.19298802316188812, "cos_base_base_plus_delta": 0.9857959764024304, "mean_kl_pi_edge_to_mean_pi": 0.18854598605548645, "mean_router_entropy": 0.8241475633578591, "operator_output_norm": [1.8197855949401855, 3.638413190841675, 2.9733684062957764, 1.6604077816009521], "orthogonal_fraction": 0.9575132131576538, "orthogonal_norm_ratio": 0.18478858470916748, "pairwise_operator_output_cosine": [0.13469204306602478, 0.584097683429718, 0.6260624527931213, 0.4268251061439514, 0.544226348400116, 0.7071696519851685], "parallel_norm_ratio": 0.05565568059682846, "per_operator_across_edge_routing_std": [0.23484232116438755, 0.09665582433226826, 0.1534598160274163, 0.04997083821366308], "per_operator_mean_routing_probability": [0.6433862290673372, 0.05768068996612697, 0.12575813252301016, 0.17317494845904238]}, "visual": {"correction_norm_ratio": 0.4115464389324188, "cos_base_base_plus_delta": 0.9384730023314964, "mean_kl_pi_edge_to_mean_pi": 0.32747069079896807, "mean_router_entropy": 0.6891108681339199, "operator_output_norm": [3.6439969539642334, 4.053160190582275, 3.9203789234161377, 6.1878275871276855], "orthogonal_fraction": 0.9592972993850708, "orthogonal_norm_ratio": 0.39479538798332214, "pairwise_operator_output_cosine": [0.7284285426139832, 0.814433217048645, 0.34904196858406067, 0.8150795102119446, 0.5788477063179016, 0.44763076305389404], "parallel_norm_ratio": 0.11621982604265213, "per_operator_across_edge_routing_std": [0.0884459335600163, 0.1385826749855661, 0.1333128480139326, 0.3485197121762939], "per_operator_mean_routing_probability": [0.08977392066062707, 0.14472255293128425, 0.1104444696956692, 0.6550590569478345]}}`
- Matched shuffle: `{"edge_count": 283080, "method": "deterministic cyclic shift within target-degree/source-degree/normalized-weight quantile bins", "non_singleton_bin_count": 7, "preserves_edge_marginal_within_each_bin": true, "quantile_bins_per_feature": 4, "quantile_cutpoints": {"normalized_weight": [0.00704225292429328, 0.009174310602247715, 0.02777777425944805], "source_degree": [35.0, 108.0, 141.0], "target_degree": [35.0, 108.0, 141.0]}, "shuffled_edge_count": 283080, "singleton_edge_count": 0, "structural_bin_count": 7}`

## Reddit-S / seed 43 / p0_operator_routed

- normal: Acc=0.96288139, Macro-F1=0.93178525, CE=0.15102662
- router_mean: Acc=0.96256685, Macro-F1=0.93127133, CE=0.15113847
- router_uniform: Acc=0.96130860, Macro-F1=0.92801582, CE=0.15636662
- router_matched_shuffle: Acc=0.96256685, Macro-F1=0.93082776, CE=0.15004896
- delta_off: Acc=0.96067947, Macro-F1=0.92448963, CE=0.15111062
- delta_parallel_only: Acc=0.96036488, Macro-F1=0.92370711, CE=0.15102839
- delta_orthogonal_only: Acc=0.96288139, Macro-F1=0.93259075, CE=0.14983484
- Routed diagnostics: `{"text": {"correction_norm_ratio": 0.22605407238006592, "cos_base_base_plus_delta": 0.9786924191041402, "mean_kl_pi_edge_to_mean_pi": 0.0009696546929374161, "mean_router_entropy": 1.3847632085408594, "operator_output_norm": [2.093754529953003, 2.4877939224243164, 2.3549904823303223, 3.378688335418701], "orthogonal_fraction": 0.983955442905426, "orthogonal_norm_ratio": 0.2224271297454834, "pairwise_operator_output_cosine": [0.7801912426948547, 0.8481922745704651, 0.6677061319351196, 0.8172225952148438, 0.7304956316947937, 0.7016085386276245], "parallel_norm_ratio": 0.04033128544688225, "per_operator_across_edge_routing_std": [0.012546023465765117, 0.01317233345543634, 0.008806085364524817, 0.008364129411800975], "per_operator_mean_routing_probability": [0.2425489463983821, 0.24421406245377333, 0.24931001939006878, 0.2639269716520232]}, "visual": {"correction_norm_ratio": 0.33341270685195923, "cos_base_base_plus_delta": 0.9557058737812633, "mean_kl_pi_edge_to_mean_pi": 0.19493298578142393, "mean_router_entropy": 0.566054896708866, "operator_output_norm": [4.154929161071777, 4.344610691070557, 4.5078253746032715, 4.069156169891357], "orthogonal_fraction": 0.9684717059135437, "orthogonal_norm_ratio": 0.32290077209472656, "pairwise_operator_output_cosine": [0.5392499566078186, 0.5427550077438354, 0.7725479006767273, 0.6141987442970276, 0.6453918218612671, 0.7458788156509399], "parallel_norm_ratio": 0.08306090533733368, "per_operator_across_edge_routing_std": [0.03443645833290106, 0.2563912494913171, 0.15986327725993274, 0.06458566127055466], "per_operator_mean_routing_probability": [0.02971408793183844, 0.7563951890550841, 0.1646109704510968, 0.049279751724273825]}}`
- Matched shuffle: `{"edge_count": 283080, "method": "deterministic cyclic shift within target-degree/source-degree/normalized-weight quantile bins", "non_singleton_bin_count": 7, "preserves_edge_marginal_within_each_bin": true, "quantile_bins_per_feature": 4, "quantile_cutpoints": {"normalized_weight": [0.00704225292429328, 0.009174310602247715, 0.02777777425944805], "source_degree": [35.0, 108.0, 141.0], "target_degree": [35.0, 108.0, 141.0]}, "shuffled_edge_count": 283080, "singleton_edge_count": 0, "structural_bin_count": 7}`

## Reddit-S / seed 44 / p0_operator_routed

- normal: Acc=0.96067947, Macro-F1=0.92874341, CE=0.17337224
- router_mean: Acc=0.95942122, Macro-F1=0.92822071, CE=0.17116228
- router_uniform: Acc=0.95973575, Macro-F1=0.92562588, CE=0.17465007
- router_matched_shuffle: Acc=0.96099401, Macro-F1=0.93045249, CE=0.16924112
- delta_off: Acc=0.95784837, Macro-F1=0.92117100, CE=0.17300983
- delta_parallel_only: Acc=0.95910662, Macro-F1=0.92460447, CE=0.17136711
- delta_orthogonal_only: Acc=0.96005034, Macro-F1=0.92813072, CE=0.17316453
- Routed diagnostics: `{"text": {"correction_norm_ratio": 0.2610257565975189, "cos_base_base_plus_delta": 0.9780667125900805, "mean_kl_pi_edge_to_mean_pi": 0.1428857413175697, "mean_router_entropy": 1.2197331398955555, "operator_output_norm": [3.501791477203369, 1.7754814624786377, 1.8718641996383667, 2.43799090385437], "orthogonal_fraction": 0.9537473917007446, "orthogonal_norm_ratio": 0.24895262718200684, "pairwise_operator_output_cosine": [0.33887210488319397, 0.44304803013801575, 0.8543559908866882, 0.6848954558372498, 0.3994561433792114, 0.4985765814781189], "parallel_norm_ratio": 0.07846669852733612, "per_operator_across_edge_routing_std": [0.24548383460191917, 0.10214906735513375, 0.08271072598093133, 0.0655623659423682], "per_operator_mean_routing_probability": [0.3473439105549637, 0.22487090258138914, 0.21416820321711186, 0.2136169837815235]}, "visual": {"correction_norm_ratio": 0.31071701645851135, "cos_base_base_plus_delta": 0.9597306856718949, "mean_kl_pi_edge_to_mean_pi": 0.16211907368329292, "mean_router_entropy": 0.9435672263571129, "operator_output_norm": [3.5294010639190674, 4.926177024841309, 5.821025371551514, 3.4596786499023438], "orthogonal_fraction": 0.9555760622024536, "orthogonal_norm_ratio": 0.2969137728214264, "pairwise_operator_output_cosine": [0.7788487076759338, 0.6091120839118958, 0.590419352054596, 0.7363494038581848, 0.6296787261962891, 0.6920384764671326], "parallel_norm_ratio": 0.09158217906951904, "per_operator_across_edge_routing_std": [0.0815754183870827, 0.10011055280637425, 0.10424409808696618, 0.17723299251117886], "per_operator_mean_routing_probability": [0.078686706521219, 0.09990462925527427, 0.26152124598101556, 0.5598874186541082]}}`
- Matched shuffle: `{"edge_count": 283080, "method": "deterministic cyclic shift within target-degree/source-degree/normalized-weight quantile bins", "non_singleton_bin_count": 7, "preserves_edge_marginal_within_each_bin": true, "quantile_bins_per_feature": 4, "quantile_cutpoints": {"normalized_weight": [0.00704225292429328, 0.009174310602247715, 0.02777777425944805], "source_degree": [35.0, 108.0, 141.0], "target_degree": [35.0, 108.0, 141.0]}, "shuffled_edge_count": 283080, "singleton_edge_count": 0, "structural_bin_count": 7}`
