
(function(){
const ROWS = [["material_type", 33556, 0.571, 0.6988, 22.4, "*", 0.7245, 26.9, "*", 0.696, 21.9, "*", -3.9, "*", "macro_f1"], ["density", 23678, 0.9898, 0.9786, -1.1, "*", 0.9812, -0.9, "\u00b7", 0.9893, -0.1, "\u00b7", 0.8, "\u00b7", "r2"], ["final_energy", 23678, 0.7739, 0.5904, -23.7, "*", 0.6042, -21.9, "*", 0.7033, -9.1, "*", 16.4, "*", "r2"], ["total_magnetization", 23678, 0.7183, 0.665, -7.4, "*", 0.6606, -8.0, "*", 0.7049, -1.9, "", 6.7, "*", "r2"], ["volume", 23678, 0.6191, 0.5192, -16.1, "*", 0.5166, -16.6, "*", 0.5767, -6.9, "*", 11.6, "*", "r2"], ["efermi", 23668, 0.9073, 0.895, -1.4, "*", 0.8941, -1.4, "*", 0.903, -0.5, "\u00b7", 1.0, "\u00b7", "r2"], ["formation_energy", 23180, 0.9947, 0.9812, -1.4, "*", 0.9845, -1.0, "*", 0.9945, -0.0, "", 1.0, "*", "r2"], ["seebeck", 8072, 0.7133, 0.6805, -3.6, "*", 0.7054, -1.1, "", 0.715, 0.2, "", 1.4, "", "r2"], ["tc", 7207, 0.8153, 0.8021, -1.6, "*", 0.7999, -1.9, "*", 0.8117, -0.4, "", 1.5, "*", "r2"], ["dos_density", 7009, 0.625, 0.5689, -9.0, "*", 0.5732, -8.3, "*", 0.5988, -4.2, "*", 4.5, "*", "r2"], ["curie", 6272, 0.8027, 0.7581, -5.6, "*", 0.7676, -4.4, "*", 0.7906, -1.5, "", 3.0, "*", "r2"], ["electrical_resistivity", 5051, 0.1767, 0.1738, -1.6, "", 0.1919, 8.6, "*", 0.1877, 6.2, "", -2.2, "", "r2"], ["thermal_conductivity", 4272, 0.7194, 0.6993, -2.8, "*", 0.7197, 0.0, "", 0.7187, -0.1, "", -0.1, "", "r2"], ["kp", 3875, 0.6957, 0.6762, -2.8, "*", 0.691, -0.7, "", 0.6858, -1.4, "", -0.8, "", "r2"], ["klat", 3863, 0.722, 0.7223, 0.0, "", 0.7123, -1.3, "", 0.7155, -0.9, "", 0.4, "", "r2"], ["power_factor", 3638, 0.6941, 0.6775, -2.3, "*", 0.7134, 2.8, "*", 0.7049, 1.5, "", -1.2, "", "r2"], ["neel", 3466, 0.7176, 0.6959, -3.0, "*", 0.6925, -3.5, "*", 0.7244, 0.9, "", 4.6, "*", "r2"], ["zt", 3445, 0.66, 0.6809, 3.2, "", 0.7031, 6.5, "*", 0.7045, 6.7, "*", 0.2, "", "r2"], ["dielectric_total", 3124, 0.6587, 0.6369, -3.3, "*", 0.6525, -0.9, "", 0.6808, 3.4, "*", 4.3, "*", "r2"], ["dielectric_ionic", 3124, 0.591, 0.5668, -4.1, "*", 0.568, -3.9, "*", 0.6076, 2.8, "", 7.0, "*", "r2"], ["dielectric_electronic", 3124, 0.8574, 0.8359, -2.5, "*", 0.8435, -1.6, "*", 0.8658, 1.0, "", 2.6, "*", "r2"], ["magnetization", 1160, 0.7611, 0.7742, 1.7, "", 0.773, 1.6, "", 0.7983, 4.9, "*", 3.3, "*", "r2"], ["magnetic_moment", 851, 0.698, 0.6538, -6.3, "*", 0.6588, -5.6, "*", 0.7025, 0.7, "", 6.6, "*", "r2"], ["magnetic_susceptibility", 58, 0.1712, -0.0773, -145.1, "*", 0.0945, -44.8, "", -0.0094, -105.5, "*", -109.9, "*", "r2"]];
const COUNTS = {"xfer_vs_single": {"better": 1, "worse": 19, "unresolved": 4, "better (negligible)": 0, "worse (negligible)": 0}, "ftz_vs_single": {"better": 4, "worse": 12, "unresolved": 7, "better (negligible)": 0, "worse (negligible)": 1}, "ftf_vs_single": {"better": 4, "worse": 4, "unresolved": 14, "better (negligible)": 0, "worse (negligible)": 2}, "ftf_vs_xfer": {"better": 19, "worse": 0, "unresolved": 2, "better (negligible)": 3, "worse (negligible)": 0}, "ftf_vs_ftz": {"better": 13, "worse": 2, "unresolved": 7, "better (negligible)": 2, "worse (negligible)": 0}};
const REP = {"zt": {"single": [0.6633331846599427, 0.68093606392038, 0.6310288929727408, 0.6991981696756027, 0.6254813312639369], "xfer": [0.6708077297518416, 0.6706752085901747, 0.677557708282482, 0.6713749710914103, 0.6721112509310214, 0.664996707889882, 0.6774210257850113, 0.670477087675308, 0.6725564697723447, 0.6608698881850765], "ftz": [0.7170483046200455, 0.7116637017852436, 0.7138321321259303, 0.6938099878528958, 0.7146002559494886, 0.681703317022033, 0.6965261235298568, 0.6990019531086978, 0.7002477104419739, 0.7027749079267607], "ftf": [0.7080356533110246, 0.7033863361183507, 0.7126932154029102, 0.7086461735781187, 0.7140784269476691, 0.685420314176296, 0.6904103348624362, 0.717324486935733, 0.7127948741506676, 0.6925502455418229], "epochs": {"single": 72, "xfer": 58, "ftz": 58.5, "ftf": 81.0}}, "magnetic_moment": {"single": [0.6978489516308797, 0.7006920564449897, 0.7068059892921555, 0.6962131586669532, 0.6882134993145275], "xfer": [0.678242826514799, 0.6073896834247197, 0.6601907980337984, 0.6213315443982675, 0.6572724108509975, 0.6616248497122059, 0.6264817399217326, 0.6648333877664772, 0.6774208677982443, 0.6583967336075732], "ftz": [0.6805933324413753, 0.6072982707425125, 0.7041127556372586, 0.6199901947672843, 0.6751108650932798, 0.647652209429443, 0.610096117505013, 0.6781482542340453, 0.6759064433783262, 0.6892627412085337], "ftf": [0.7333196106924573, 0.6815631297494357, 0.7259192726595817, 0.655544580510081, 0.7150229369331851, 0.675512346598348, 0.6855829159827622, 0.7032385239208566, 0.7326316364783663, 0.7167824773568954], "epochs": {"single": 142, "xfer": 56.5, "ftz": 62.5, "ftf": 80.0}}, "final_energy": {"single": [0.781923343086146, 0.7858104641616527, 0.7689518507480217, 0.7626466045333128, 0.7702093667656715], "xfer": [0.607821678761546, 0.5522402061508644, 0.5910363667540026, 0.6165311825808311, 0.6122169436199484, 0.6101530697999035, 0.6057967871069923, 0.6062817410078081, 0.6150847326585338, 0.5727218984983474], "ftz": [0.6044094000144278, 0.5721345794828947, 0.6010316466982109, 0.6220965247918165, 0.6121784846615957, 0.6071131341372291, 0.6001243271603447, 0.6097054408322095, 0.6163505895009631, 0.5970557098346088], "ftf": [0.7155216702332803, 0.6731180017313221, 0.687660529651604, 0.7193720705587272, 0.7083214331285421, 0.7238557274360589, 0.688020617935788, 0.7009195224974275, 0.7152867043483551, 0.7012838696272388], "epochs": {"single": 137, "xfer": 63, "ftz": 51.5, "ftf": 118.5}}, "material_type": {"single": [0.5761431651234872, 0.5535523360143494, 0.5940422954062614, 0.6035300810282508, 0.5275751481935703], "xfer": [0.6159266687622197, 0.7315480183767165, 0.7389104296933121, 0.5960661890910082, 0.7186706322794187, 0.6878842666095228, 0.5851018469584194, 0.6486368798449036, 0.6118932745444126, 0.49863019438683553], "ftz": [0.6946975228486865, 0.7386907702965896, 0.7022006225133384, 0.7405424285565804, 0.7564650512120036, 0.720704888389851, 0.7224257571550112, 0.7278426441636486, 0.7274265303193058, 0.7136936118777315], "ftf": [0.6314348696927401, 0.7284969011088414, 0.7207586726003744, 0.6641562808140569, 0.7211541101314609, 0.7367585057358564, 0.7182509798949848, 0.7029548265551271, 0.680783095303488, 0.6556793273720795], "epochs": {"single": 91, "xfer": 56.5, "ftz": 33.0, "ftf": 49.0}}};
const POS = {"final_energy": {"early": -13.685339166164981, "late": -23.32996261135844}, "magnetic_moment": {"early": -0.7732148617733339, "late": -5.609129892086218}, "zt": {"early": 1.9944003625871582, "late": 1.6341695918316363}};
const CLASSES=["DAC","DQC","IAC","IQC","others"];
const CM={single:{runs:5,m:[[5,0,0,0,0],[0,15,0,0,0],[0,0,114,6,0],[4,8,10,118,0],[38,24,253,107,36068]]},multi:{runs:3,m:[[3,0,0,0,0],[0,9,0,0,0],[0,0,69,3,0],[1,6,6,69,2],[7,8,54,39,21786]]}};
const BOX_MT={single:{n:5,lo:0.5276,q1:0.5536,med:0.5761,q3:0.5940,hi:0.6035},
 train:[{label:"Slots 1–6",n:50,lo:0.5362,q1:0.5853,med:0.6209,q3:0.6451,hi:0.7322},{label:"Slots 7–12",n:71,lo:0.5432,q1:0.6192,med:0.6494,q3:0.6809,hi:0.7645},{label:"Slots 13–18",n:51,lo:0.5666,q1:0.6362,med:0.6744,q3:0.6936,hi:0.7394},{label:"Slots 19–24",n:34,lo:0.5621,q1:0.6158,med:0.6502,q3:0.6963,hi:0.7546}],
 end:[{label:"Slots 1–6",n:15,lo:0.5975,q1:0.6485,med:0.6722,q3:0.6858,hi:0.7331},{label:"Slots 7–12",n:21,lo:0.5903,q1:0.6416,med:0.6580,q3:0.6936,hi:0.7479},{label:"Slots 13–18",n:15,lo:0.5829,q1:0.6314,med:0.6533,q3:0.6856,hi:0.7283},{label:"Slots 19–24",n:21,lo:0.5759,q1:0.6274,med:0.6737,q3:0.6976,hi:0.7550}]};
const EV = {"zt": {"single": 0.6599955284985206, "single_sd": 0.031689070797737945, "single_n": 5, "xfer": {"mean": 0.6790170887637796, "rel": 2.8820741117038744, "se": 0.01423100276804762, "sep": false, "n": 10}, "ftz": {"mean": 0.7031208394362926, "sd": 0.011193727623545747, "n": 10, "rel": 6.534182289974198, "se": 0.01460716931564164, "sep": true, "matters": true}, "ftf": {"mean": 0.7045340061025029, "sd": 0.011197308371846626, "n": 10, "rel": 6.748299902168528, "se": 0.014607443755915413, "sep": true, "matters": true}, "ftf_vs_ftz": {"n": 10, "mean": 0.7045340061025029, "sd": 0.011197308371846626, "delta": 0.0014131666662103859, "relative_pct": 0.2009848929158966, "se_of_difference": 0.005006787921256079, "separated": false, "practically_significant": false, "matters": false}, "ftf_vs_xfer": {"n": 10, "mean": 0.7045340061025029, "sd": 0.011197308371846626, "delta": 0.023657077587672992, "relative_pct": 3.4745012787076286, "se_of_difference": 0.0035408998118306957, "separated": true, "practically_significant": true, "matters": true}, "by_position": [[1, 14, 0.6620474097127662, 0.028700749280621905], [2, 10, 0.6747602651255666, 0.016640935056368905], [3, 6, 0.6916197559577082, 0.009060266869248582], [4, 8, 0.6759001403489269, 0.02096773202203498], [5, 5, 0.6807740534075455, 0.013128372275708625], [6, 15, 0.6741901570450708, 0.01455384546811341], [7, 5, 0.6817002527665827, 0.01624543794521272], [8, 10, 0.6642159232653894, 0.01050738109800731], [9, 11, 0.6734263800209281, 0.015950884849880304], [10, 10, 0.6712500313496531, 0.008358800892320531], [11, 11, 0.6774502487641998, 0.012748530834394369], [12, 10, 0.6697152674815484, 0.018424919036219557], [13, 7, 0.6774354309342699, 0.016987648952550734], [14, 12, 0.6735306635693864, 0.014143116557212369], [15, 20, 0.6732932717475062, 0.008682673165485174], [16, 10, 0.6777097172120818, 0.012150438569937612], [17, 9, 0.6719717016504979, 0.006741083457399349], [18, 8, 0.6709873658430404, 0.014540653041745191], [19, 6, 0.6731533117006706, 0.010483425281786776], [20, 10, 0.672022209871134, 0.01218622738230022], [21, 8, 0.6666932734494944, 0.011190318065730556], [22, 14, 0.6700432577379415, 0.010163390289738814], [23, 11, 0.6710516386626121, 0.005564667207482635], [24, 10, 0.6708848047954552, 0.005021875114066448]], "early": {"n": 73, "mean": 0.6731584817119541, "sd": 0.019622279521240356, "delta": 0.013162953213433526, "relative_pct": 1.9944003625871582, "se_of_difference": 0.014356666672860634, "separated": false, "practically_significant": true, "matters": false}, "late": {"n": 76, "mean": 0.6707809747326919, "sd": 0.009444647461112891, "delta": 0.010785446234171325, "relative_pct": 1.6341695918316363, "se_of_difference": 0.014213132792759249, "separated": false, "practically_significant": true, "matters": false}, "ft_source": {"ftz": "original", "ftf": "original"}, "probe": {"single": 0.6599955284985206, "multi": 0.7052175426451375, "rel": 6.851866746657547, "se": 0.014273617146425874, "sep": true}}, "material_type": {"single": 0.5709686051531838, "single_sd": 0.030845337984718, "single_n": 5, "xfer": {"mean": 0.6475292970649098, "rel": 13.408914469331584, "se": 0.026958281996826605, "sep": true, "n": 10}, "ftz": {"mean": 0.7244689827332746, "sd": 0.018321703778141202, "n": 10, "rel": 26.88420627591399, "se": 0.01496179995895189, "sep": true, "matters": true}, "ftf": {"mean": 0.696042756920901, "sd": 0.03581817084942787, "n": 10, "rel": 21.905609281995698, "se": 0.01784884061719914, "sep": true, "matters": true}, "ftf_vs_ftz": {"n": 10, "mean": 0.696042756920901, "sd": 0.03581817084942787, "delta": -0.02842622581237364, "relative_pct": -3.9237326220823494, "se_of_difference": 0.012722524090496972, "separated": true, "practically_significant": true, "matters": true}, "ftf_vs_xfer": {"n": 10, "mean": 0.696042756920901, "sd": 0.03581817084942787, "delta": -0.0027793870747733207, "relative_pct": -0.39772452814410597, "se_of_difference": 0.011326700150524002, "separated": false, "practically_significant": false, "matters": false}, "by_position": [[1, 11, 0.5681371732865758, 0.026131688987990346], [2, 6, 0.6038361230245484, 0.04087656264249247], [3, 13, 0.6359154165654234, 0.08558425496880145], [4, 9, 0.6274516233488381, 0.05056204250568209], [5, 6, 0.6304747680863709, 0.013439236550348543], [6, 5, 0.6256081913173547, 0.03842073589307313], [7, 10, 0.644803547748784, 0.030061172768506825], [8, 13, 0.6723155370542873, 0.04353289017598937], [9, 7, 0.6581664060047928, 0.04287170286317579], [10, 14, 0.6231759700944783, 0.05107465625577447], [11, 13, 0.6676251891869378, 0.04389705586217403], [12, 14, 0.6497773458124588, 0.04897971376782532], [13, 10, 0.678125005666543, 0.03624926839686544], [14, 8, 0.6758995209576004, 0.02369277264939156], [15, 9, 0.66793004066461, 0.04734598114596222], [16, 9, 0.6699054083671628, 0.03482336298608283], [17, 5, 0.6205995358677157, 0.04415038294058927], [18, 10, 0.6226747720642518, 0.08350335211449501], [19, 14, 0.6437937892836596, 0.09048383358935463], [20, 10, 0.6433859415204276, 0.040060068532501894], [21, 13, 0.6630568267447029, 0.05519893808833912], [22, 9, 0.6678514576357955, 0.03780049260091522], [23, 12, 0.635357728710714, 0.06786703206132912], [24, 10, 0.6433268400546769, 0.07667327751856019]], "early": {"n": 73, "mean": 0.6285687355354553, "sd": 0.056929372851996646, "delta": 0.05760013038227152, "relative_pct": 10.08814317677206, "se_of_difference": 0.01531938635437087, "separated": true, "practically_significant": true, "matters": true}, "late": {"n": 83, "mean": 0.6441527816462077, "sd": 0.06635356416929453, "delta": 0.07318417649302389, "relative_pct": 12.817548256158407, "se_of_difference": 0.01559912509496999, "separated": true, "practically_significant": true, "matters": true}, "ft_source": {"ftz": "original", "ftf": "original"}}, "magnetic_moment": {"single": 0.6979547310699011, "single_sd": 0.006778155762492842, "single_n": 5, "xfer": {"mean": 0.6578317465762104, "rel": -5.748651410699072, "se": 0.008844505387189014, "sep": true, "n": 10}, "ftz": {"mean": 0.6588171184437072, "sd": 0.035030416948584095, "n": 10, "rel": -5.6074714997919095, "se": 0.011484846114218848, "sep": true, "matters": true}, "ftf": {"mean": 0.7025117430881969, "sd": 0.026719257563295084, "n": 10, "rel": 0.6529093959016855, "se": 0.008976667064200005, "sep": false, "matters": false}, "ftf_vs_ftz": {"n": 10, "mean": 0.7025117430881969, "sd": 0.026719257563295084, "delta": 0.0436946246444897, "relative_pct": 6.632284350429065, "se_of_difference": 0.01393215287141707, "separated": true, "practically_significant": true, "matters": true}, "ftf_vs_xfer": {"n": 10, "mean": 0.7025117430881969, "sd": 0.026719257563295084, "delta": 0.04871248204221157, "relative_pct": 7.450678663092791, "se_of_difference": 0.008449371128869305, "separated": true, "practically_significant": true, "matters": true}, "by_position": [[1, 11, 0.7036885062722269, 0.00984112191888942], [2, 8, 0.7021585544883533, 0.029793642385383044], [3, 12, 0.6844057662182398, 0.025227320982146566], [4, 12, 0.6830866253097857, 0.03628950963851311], [5, 12, 0.6981441401728895, 0.020423435682122532], [6, 6, 0.6760030270709713, 0.03004335991249621], [7, 10, 0.6959457702992248, 0.013568622242006119], [8, 8, 0.6938911270967866, 0.01899046441727527], [9, 11, 0.6809758715716133, 0.027846413028396515], [10, 9, 0.676673001100708, 0.032490024352017935], [11, 10, 0.6793399198976399, 0.03733365672802745], [12, 11, 0.6727101287690577, 0.02888754315257246], [13, 12, 0.6650534456434476, 0.032888530717670554], [14, 9, 0.6731877444004665, 0.02167651782890842], [15, 6, 0.6698535059573872, 0.02708567724454467], [16, 9, 0.6558481195436219, 0.06612170345675354], [17, 12, 0.6642429066580243, 0.02671591475821864], [18, 9, 0.6437255427156836, 0.028909483613126324], [19, 12, 0.6568580404188351, 0.026063840589999318], [20, 10, 0.6717243857953648, 0.02005080428384867], [21, 9, 0.6530969727231928, 0.03097346687078639], [22, 8, 0.6629235337097692, 0.018388300710441247], [23, 14, 0.6629453350164605, 0.030068971006012697], [24, 10, 0.6513184842028815, 0.024278973984486737]], "early": {"n": 79, "mean": 0.6925580413608186, "sd": 0.0248259638892239, "delta": -0.0053966897090825805, "relative_pct": -0.7732148617733339, "se_of_difference": 0.004121929824193918, "separated": false, "practically_significant": false, "matters": false}, "late": {"n": 84, "mean": 0.6588055436162293, "sd": 0.0264516537412738, "delta": -0.0391491874536718, "relative_pct": -5.609129892086218, "se_of_difference": 0.004185489433078479, "separated": true, "practically_significant": true, "matters": true}, "ft_source": {"ftz": "original", "ftf": "original"}}, "final_energy": {"single": 0.773908325858961, "single_sd": 0.009630377828053023, "single_n": 5, "xfer": {"mean": 0.6031086548427591, "rel": -22.069754945022886, "se": 0.00785926191628586, "sep": true, "n": 10}, "ftz": {"mean": 0.60421998371143, "sd": 0.013647532838187149, "n": 10, "rel": -21.92615539562697, "se": 0.006097077224305979, "sep": true, "matters": true}, "ftf": {"mean": 0.7033360147148343, "sd": 0.016322576492960674, "n": 10, "rel": -9.118949723896359, "se": 0.006722461287286546, "sep": true, "matters": true}, "ftf_vs_ftz": {"n": 10, "mean": 0.7033360147148343, "sd": 0.016322576492960674, "delta": 0.09911603100340427, "relative_pct": 16.403964396308545, "se_of_difference": 0.0067281621260040165, "separated": true, "practically_significant": true, "matters": true}, "ftf_vs_xfer": {"n": 10, "mean": 0.7033360147148343, "sd": 0.016322576492960674, "delta": 0.11293116138116854, "relative_pct": 19.127749499942844, "se_of_difference": 0.005161651900007907, "separated": true, "practically_significant": true, "matters": true}, "by_position": [[1, 11, 0.7758212440000084, 0.009120756465019118], [2, 4, 0.6828532893076522, 0.027911384888155488], [3, 11, 0.6630399127417949, 0.020311179125807172], [4, 7, 0.6626602701266634, 0.03581760973830903], [5, 6, 0.6444927528608354, 0.022293059806853036], [6, 8, 0.6322416719194195, 0.023548574089108563], [7, 12, 0.6253294615539039, 0.015002910317123287], [8, 5, 0.6250829961747921, 0.01289208238559476], [9, 12, 0.6152074171105698, 0.015424123185294672], [10, 17, 0.6096338898336523, 0.011673512979320871], [11, 10, 0.6070148285161456, 0.023401939470051907], [12, 10, 0.6074535309748409, 0.016639670903010464], [13, 10, 0.5948422789318449, 0.019007455077887585], [14, 10, 0.6009569318202005, 0.01814179558243105], [15, 16, 0.5948161765238992, 0.018779802905844724], [16, 7, 0.5818428661018811, 0.018330287500012362], [17, 10, 0.5871573281162503, 0.010171640794850867], [18, 13, 0.5815019302379177, 0.014103566625582731], [19, 15, 0.5988364006066867, 0.021450528462285078], [20, 14, 0.5940142940330221, 0.011563789450771762], [21, 7, 0.6109105205154056, 0.029641078153339197], [22, 8, 0.5844374668637922, 0.012421819415449975], [23, 7, 0.5957549927249325, 0.014044232989958492], [24, 10, 0.5989884606938778, 0.021035992165581198]], "early": {"n": 64, "mean": 0.6679963466299729, "sd": 0.056108607076981336, "delta": -0.1059119792289881, "relative_pct": -13.685339166164981, "se_of_difference": 0.00823037557536542, "separated": true, "practically_significant": true, "matters": true}, "late": {"n": 84, "mean": 0.5933558027898753, "sd": 0.01863876676046297, "delta": -0.18055252306908565, "relative_pct": -23.32996261135844, "se_of_difference": 0.004762834542451569, "separated": true, "practically_significant": true, "matters": true}, "ft_source": {"ftz": "original", "ftf": "original"}}};
const KIND = [["material_type", "classification", 33556, 21.905609281995698, true, true, 0.01784884061719914], ["density", "intensive", 23678, -0.05065693170054142, false, true, 0.00023815395813816073], ["final_energy", "extensive", 23678, -9.118949723896359, true, true, 0.006722461287286546], ["total_magnetization", "extensive", 23678, -1.8657515012598258, false, false, 0.010142228534424954], ["volume", "extensive", 23678, -6.8553314274179105, true, true, 0.0058185848813734845], ["efermi", "intensive", 23668, -0.46933637762377917, false, true, 0.0013311380350663174], ["formation_energy", "intensive", 23180, -0.019929586520385734, false, false, 0.000214049161583141], ["seebeck", "intensive", 8072, 0.2369827670092127, false, false, 0.00579772456978732], ["tc", "intensive", 7207, -0.43726242789030684, false, false, 0.0030302867842430933], ["dos_density", "intensive", 7009, -4.1949169947753076, true, true, 0.003023613712519873], ["curie", "intensive", 6272, -1.5020382992585768, false, false, 0.007036667583950864], ["electrical_resistivity", "intensive", 5051, 6.2364999076058805, false, false, 0.006862999309698158], ["thermal_conductivity", "intensive", 4272, -0.08458795937207514, false, false, 0.006643162844539942], ["kp", "intensive", 3875, -1.423365202923367, false, false, 0.006785782826727729], ["klat", "intensive", 3863, -0.9009043738337558, false, false, 0.00653539548599939], ["power_factor", "intensive", 3638, 1.5494522185786788, false, false, 0.006647221560220371], ["neel", "intensive", 3466, 0.9444982928599839, false, false, 0.00743953974494966], ["zt", "intensive", 3445, 6.748299902168528, true, true, 0.014607443755915413], ["dielectric_total", "intensive", 3124, 3.351923390542868, true, true, 0.007169022284151022], ["dielectric_ionic", "intensive", 3124, 2.8085574199236776, false, false, 0.011292566134721968], ["dielectric_electronic", "intensive", 3124, 0.9756467899978449, false, false, 0.00461039191529302], ["magnetization", "intensive", 1160, 4.881777502522171, true, true, 0.012084897414355554], ["magnetic_moment", "intensive", 851, 0.6529093959016855, false, false, 0.008976667064200005]];
const SHARE = {"zt": 4.2, "magnetic_moment": 1.1, "final_energy": 24.4, "material_type": 32.2};
const NS="http://www.w3.org/2000/svg";
const el=(n,a)=>{const e=document.createElementNS(NS,n);for(const k in a)e.setAttribute(k,a[k]);return e;};
const css=v=>getComputedStyle(document.documentElement).getPropertyValue(v).trim();
const pretty=s=>s.replace(/_/g," ");
const med=a=>{const s=a.slice().sort((x,y)=>x-y);const n=s.length;return n%2?s[(n-1)/2]:(s[n/2-1]+s[n/2])/2;};

// ---- stats tiles ----
const c=k=>`${COUNTS[k].better} / ${COUNTS[k].worse} / ${COUNTS[k].unresolved}`;
document.getElementById("s-xfer").textContent=c("xfer_vs_single");
document.getElementById("s-frz").textContent=c("ftz_vs_single");
document.getElementById("s-warm").textContent=c("ftf_vs_single");
document.getElementById("s-rec").textContent=c("ftf_vs_xfer");

// ---- share table ----
const N={zt:3445,magnetic_moment:851,final_energy:23678,material_type:33556};
const st=document.getElementById("sharetable");
for(const t of ["magnetic_moment","zt","final_energy","material_type"]){
  const tr=document.createElement("tr");
  const rep=Math.round(N[t]/(SHARE[t]/100)-N[t]);
  tr.innerHTML=`<td>${pretty(t)}</td><td>${N[t].toLocaleString()}</td><td>${rep.toLocaleString()}</td><td class="${SHARE[t]<5?'neg':''}">${SHARE[t]}%</td>`;
  st.appendChild(tr);
}

// ---- big table ----
const tb=document.querySelector("#bigtable tbody");
const fmtp=(p,m)=>p==null?"-":`${p>0?"+":""}${p.toFixed(1)}%${m||""}`;
const cls=(p,m)=>(!m||m==="·")?"":(p>0?"pos":"neg");
for(const r of ROWS){
  const [task,n,single,xf,xfp,xfm,fz,fzp,fzm,wm,wmp,wmm,wf,wfm,metric]=r;
  const tr=document.createElement("tr");
  tr.innerHTML=`<td>${pretty(task)}${metric==="macro_f1"?" <span style='color:var(--muted);font-size:11px'>(macro-F1)</span>":""}</td><td>${n.toLocaleString()}</td><td>${single.toFixed(4)}</td>
   <td>${xf==null?"-":xf.toFixed(4)}</td><td class="${cls(xfp,xfm)}">${fmtp(xfp,xfm)}</td>
   <td>${fz.toFixed(4)}</td><td class="${cls(fzp,fzm)}">${fmtp(fzp,fzm)}</td>
   <td>${wm.toFixed(4)}</td><td class="${cls(wmp,wmm)}">${fmtp(wmp,wmm)}</td>
   <td class="${cls(wf,wfm)}">${fmtp(wf,wfm)}</td>`;
  tb.appendChild(tr);
}

function draw(){
  const C={xfer:css("--xfer"),frz:css("--frz"),warm:css("--warm"),alone:css("--alone")};
  const cRule=css("--rule"),cSoft=css("--rule-soft"),cInk=css("--ink"),cMuted=css("--muted"),cSurf=css("--surface");

  // ---- fig0: schematic ----
  try{(function(){
    const svg=document.getElementById("fig0"); svg.textContent="";
    const box=(x,y,w,h,label,sub,col,dash)=>{
      svg.appendChild(el("rect",{x,y,width:w,height:h,rx:6,fill:cSurf,stroke:col||cRule,"stroke-width":1.6,...(dash?{"stroke-dasharray":"6 4"}:{})}));
      const t=el("text",{x:x+w/2,y:y+(sub?h/2-4:h/2+4),"text-anchor":"middle",class:"lab"}); t.setAttribute("fill",col||cInk); t.textContent=label; svg.appendChild(t);
      if(sub){const s=el("text",{x:x+w/2,y:y+h/2+13,"text-anchor":"middle",class:"axis"}); s.textContent=sub; svg.appendChild(s);}
    };
    const arrow=(x1,y1,x2,y2,col,dash)=>{
      svg.appendChild(el("line",{x1,y1,x2,y2,stroke:col,"stroke-width":1.8,...(dash?{"stroke-dasharray":"6 4"}:{})}));
      const a=Math.atan2(y2-y1,x2-x1);const p=(dx,dy)=>`${x2-dx*Math.cos(a)+dy*Math.sin(a)},${y2-dx*Math.sin(a)-dy*Math.cos(a)}`;
      svg.appendChild(el("polygon",{points:`${x2},${y2} ${p(9,4.5)} ${p(9,-4.5)}`,fill:col}));
    };
    box(20,95,190,60,"23-task encoder","continual, replay on",cInk);
    box(285,40,240,60,"step 24: add X","replay on · total-loss stop",C.xfer);
    arrow(210,110,285,72,C.xfer);
    const lx=el("text",{x:245,y:82,"text-anchor":"middle",class:"val"}); lx.setAttribute("fill",C.xfer); lx.textContent="xfer"; svg.appendChild(lx);
    box(600,20,170,48,"frozen","head only",C.frz); arrow(525,60,600,44,C.frz);
    box(600,92,170,48,"warm-start","encoder + head",C.warm); arrow(525,80,600,116,C.warm);
    const nr=el("text",{x:784,y:48,class:"axis"}); nr.textContent="no replay"; svg.appendChild(nr);
    const nr2=el("text",{x:784,y:120,class:"axis"}); nr2.textContent="no replay"; svg.appendChild(nr2);
    box(600,168,170,48,"frozen, unseen","fresh head",C.frz,true); arrow(210,135,600,185,C.frz,true);
    box(792,168,168,48,"warm-start, unseen","encoder + fresh head",C.warm,true);
    arrow(770,192,792,192,C.warm,true);
    const nb=el("text",{x:330,y:200,class:"axis"}); nb.textContent="X never seen · running"; svg.appendChild(nb);
    const al=el("text",{x:20,y:236,class:"axis"}); al.textContent="reference “alone”: a fresh model trained on X only, 5 seeds"; svg.appendChild(al);
  })();}catch(e){console.error('figure failed',e);}

  // ---- fig1: per-task dot plot ----
  try{(function(){
    const svg=document.getElementById("fig1"); svg.textContent="";
    const W=960,rowH=30,m={t:32,r:24,b:40,l:190}; const rows=ROWS.filter(r=>r[0]!=="magnetic_susceptibility").slice().sort((a,b)=>(b[10]??-99)-(a[10]??-99));
    const H=m.t+rows.length*rowH+m.b; svg.setAttribute("height",H); const iw=W-m.l-m.r;
    const lo=-30,hi=30; const X=v=>m.l+(Math.max(lo,Math.min(hi,v))-lo)/(hi-lo)*iw;
    for(let v=-30;v<=30.01;v+=10){const x=X(v);svg.appendChild(el("line",{x1:x,x2:x,y1:m.t-10,y2:m.t+rows.length*rowH,stroke:v===0?cRule:cSoft,"stroke-width":v===0?1.6:1}));
      const t=el("text",{x,y:m.t-16,"text-anchor":"middle",class:"axis"});t.textContent=(v>0?"+":"")+v+"%";svg.appendChild(t);}
    rows.forEach((r,i)=>{
      const [task,n,single,xf,xfp,xfm,fz,fzp,fzm,wm,wmp,wmm]=r; const y=m.t+i*rowH+rowH/2;
      const nm=el("text",{x:m.l-12,y:y+4,"text-anchor":"end",class:"name"}); nm.setAttribute("fill",cInk); nm.textContent=pretty(task); svg.appendChild(nm);
      if(xfp!=null&&wmp!=null) svg.appendChild(el("line",{x1:X(xfp),x2:X(wmp),y1:y,y2:y,stroke:cMuted,"stroke-width":1,opacity:.45}));
      const dot=(p,mk,col,shape)=>{ if(p==null)return; const filled=mk==="*"; const x=X(p);
        if(shape==="sq") svg.appendChild(el("rect",{x:x-5,y:y-5,width:10,height:10,fill:filled?col:cSurf,stroke:col,"stroke-width":1.8}));
        else svg.appendChild(el("circle",{cx:x,cy:y,r:5.5,fill:filled?col:cSurf,stroke:col,"stroke-width":1.8}));
        if(p<lo||p>hi){const c=el("text",{x:x+(p<lo?-10:10),y:y+4,"text-anchor":p<lo?"end":"start",class:"val"});c.setAttribute("fill",col);c.textContent=(p>0?"+":"")+p.toFixed(0)+"%";svg.appendChild(c);}
      };
      dot(xfp,xfm,C.xfer,"sq"); dot(fzp,fzm,C.frz); dot(wmp,wmm,C.warm);
    });
    const ax=el("text",{x:m.l+iw/2,y:H-10,"text-anchor":"middle",class:"axis"}); ax.textContent="change vs training alone (%)"; svg.appendChild(ax);
  })();}catch(e){console.error('figure failed',e);}


  // ---- zt arc: four measurements with 2SE ----
  try{(function(){
    const svg=document.getElementById("fig-zt-arc"); if(!svg) return; svg.textContent="";
    const E=EV.zt, base=E.single;
    const pts=[["6-task probe",E.probe.rel,E.probe.se?2*E.probe.se/base*100:null,C.alone,"n = 25 vs 5"],
               ["xfer",E.xfer.rel,E.xfer.se?2*E.xfer.se/base*100:null,C.xfer,"n = 10"],
               ["frozen",E.ftz.rel,2*E.ftz.se/base*100,C.frz,"n = 10"],
               ["warm-start",E.ftf.rel,2*E.ftf.se/base*100,C.warm,"n = 10"]];
    const W=980,H=250,m={t:26,r:20,b:48,l:120}; const iw=W-m.l-m.r,ih=H-m.t-m.b; svg.setAttribute("width",W);svg.setAttribute("height",H);
    const lo=-4,hi=18; const X=v=>m.l+(v-lo)/(hi-lo)*iw;
    for(let v=-4;v<=18.01;v+=2){svg.appendChild(el("line",{x1:X(v),x2:X(v),y1:m.t,y2:m.t+ih,stroke:v===0?cRule:cSoft,"stroke-width":v===0?1.6:1}));const tx=el("text",{x:X(v),y:m.t+ih+18,"text-anchor":"middle",class:"axis"});tx.textContent=(v>0?"+":"")+v+"%";svg.appendChild(tx);}
    const rowH=ih/pts.length;
    pts.forEach(([name,rel,two,col,nlab],i)=>{const y=m.t+rowH*i+rowH/2;
      const nm=el("text",{x:m.l-12,y:y+4,"text-anchor":"end",class:"lab"});nm.setAttribute("fill",col);nm.textContent=name;svg.appendChild(nm);
      if(two!=null){svg.appendChild(el("line",{x1:X(rel-two),x2:X(rel+two),y1:y,y2:y,stroke:col,"stroke-width":2}));[rel-two,rel+two].forEach(w=>svg.appendChild(el("line",{x1:X(w),x2:X(w),y1:y-6,y2:y+6,stroke:col,"stroke-width":2})));}
      svg.appendChild(el("circle",{cx:X(rel),cy:y,r:6,fill:col}));
      const v=el("text",{x:X(rel+(two||0))+10,y:y+4,class:"val"});v.setAttribute("fill",col);v.textContent=`${rel>0?"+":""}${rel.toFixed(1)}% ± ${two!=null?two.toFixed(1):"?"}  (${nlab})`;svg.appendChild(v);});
    const ax=el("text",{x:m.l+iw/2,y:H-6,"text-anchor":"middle",class:"axis"});ax.textContent="change vs training alone (%) · bar = ±2×SE of the difference";svg.appendChild(ax);
  })();}catch(e){console.error('figure failed',e);}

  // ---- position curves: every run as a dot, the median at each position as a bar ----
  try{document.querySelectorAll("svg.posfig").forEach(svg=>{
    svg.textContent=""; const task=svg.dataset.task; const E=EV[task]; const R=POSRUNS[task]; const alone=REP[task].single; const BM=med(alone),aLo=Math.min(...alone),aHi=Math.max(...alone);
    const W=980,H=390,m={t:40,r:150,b:60,l:70}; const iw=W-m.l-m.r,ih=H-m.t-m.b; svg.setAttribute("width",W);svg.setAttribute("height",H);
    const vals=R.map(r=>r[1]).filter(v=>v!=null); let lo=Math.min(...vals,aLo),hi=Math.max(...vals,aHi); const pad=(hi-lo)*.08; lo-=pad;hi+=pad;
    const Y=v=>m.t+ih-(v-lo)/(hi-lo)*ih, X=p=>m.l+(p-1)/23*iw;
    for(let k=0;k<=4;k++){const v=lo+(hi-lo)*k/4;svg.appendChild(el("line",{x1:m.l,x2:m.l+iw,y1:Y(v),y2:Y(v),stroke:cSoft}));const tx=el("text",{x:m.l-10,y:Y(v)+4,"text-anchor":"end",class:"axis"});tx.textContent=v.toFixed(3);svg.appendChild(tx);}
    [[1,8,E.early,"early 1–8"],[17,24,E.late,"late 17–24"]].forEach(([a,b,d,lab])=>{
      svg.appendChild(el("rect",{x:X(a)-13,y:m.t-6,width:X(b)-X(a)+26,height:ih+10,fill:C.xfer,opacity:.06}));
      const t=el("text",{x:(X(a)+X(b))/2,y:m.t+10,"text-anchor":"middle",class:"val"});t.setAttribute("fill",C.xfer);
      t.textContent=`${lab}: ${d.relative_pct>0?"+":""}${d.relative_pct.toFixed(1)}% vs alone${d.matters?" *":""}`;svg.appendChild(t);});
    svg.appendChild(el("rect",{x:m.l,y:Y(aHi),width:iw,height:Y(aLo)-Y(aHi),fill:C.alone,opacity:.14}));
    svg.appendChild(el("line",{x1:m.l,x2:m.l+iw,y1:Y(BM),y2:Y(BM),stroke:C.alone,"stroke-width":1.4,"stroke-dasharray":"5 4"}));
    const bl=el("text",{x:m.l+iw+8,y:Y(BM)+4,class:"val"});bl.setAttribute("fill",C.alone);bl.textContent="alone "+BM.toFixed(4);svg.appendChild(bl);
    const byPos={}; R.forEach(([p,v])=>{if(v!=null)(byPos[p]=byPos[p]||[]).push(v);});
    Object.keys(byPos).map(Number).sort((a,b)=>a-b).forEach(p=>{const vs=byPos[p],x=X(p);
      vs.forEach((v,k)=>{const jit=((k*37)%7-3)*2.1; svg.appendChild(el("circle",{cx:x+jit,cy:Y(v),r:2.8,fill:C.xfer,opacity:.5}));});
      const mv=med(vs); svg.appendChild(el("line",{x1:x-10,x2:x+10,y1:Y(mv),y2:Y(mv),stroke:C.xfer,"stroke-width":3}));});
    [1,4,8,12,16,20,24].forEach(p=>{const tx=el("text",{x:X(p),y:m.t+ih+22,"text-anchor":"middle",class:"axis"});tx.textContent=p;svg.appendChild(tx);});
    const xl=el("text",{x:m.l+iw/2,y:m.t+ih+44,"text-anchor":"middle",class:"axis"});xl.textContent="position of "+pretty(task)+" in the 24-task sequence";svg.appendChild(xl);
    const yl=el("text",{x:m.l,y:m.t-18,class:"axis"});yl.textContent=(task==="material_type"?"macro-F1":"R²")+" of each run at its own step";svg.appendChild(yl);
    svg.appendChild(el("line",{x1:m.l,x2:m.l+iw,y1:m.t+ih,y2:m.t+ih,stroke:cRule}));
  });}catch(e){console.error('posfig failed',e);}

  // ---- material_type by position band: every run as a dot, the median as a bar ----
  try{(function(){
    const svg=document.getElementById("fig-mt-pos"); if(!svg) return; svg.textContent="";
    const R=POSRUNS.material_type, alone=REP.material_type.single, BM=med(alone);
    const groups=[{label:"alone",strips:[{vals:alone,col:C.alone,unit:"seeds"}]}];
    [["Slots 1–6",1,6],["Slots 7–12",7,12],["Slots 13–18",13,18],["Slots 19–24",19,24]].forEach(([lab,a,b])=>{const rs=R.filter(r=>r[0]>=a&&r[0]<=b);
      groups.push({label:lab,strips:[{vals:rs.map(r=>r[1]).filter(v=>v!=null),col:C.warm,unit:"runs"},{vals:rs.map(r=>r[2]).filter(v=>v!=null),col:C.xfer,unit:"runs"}]});});
    const CL=0.45,all=groups.flatMap(g=>g.strips.flatMap(s=>s.vals)),shown=all.filter(v=>v>=CL),hidden=all.filter(v=>v<CL); let lo=Math.min(...shown),hi=Math.max(...shown); const pad=(hi-lo)*.14; lo-=pad;hi+=pad;
    const W=980,H=400,m={t:44,r:18,b:70,l:70}; const iw=W-m.l-m.r,ih=H-m.t-m.b; const Y=v=>m.t+ih-(Math.max(v,lo)-lo)/(hi-lo)*ih; svg.setAttribute("width",W);svg.setAttribute("height",H);
    for(let k=0;k<=5;k++){const v=lo+(hi-lo)*k/5;svg.appendChild(el("line",{x1:m.l,x2:m.l+iw,y1:Y(v),y2:Y(v),stroke:cSoft}));const tx=el("text",{x:m.l-10,y:Y(v)+4,"text-anchor":"end",class:"axis"});tx.textContent=v.toFixed(3);svg.appendChild(tx);}
    svg.appendChild(el("line",{x1:m.l,x2:m.l+iw,y1:Y(BM),y2:Y(BM),stroke:C.alone,"stroke-width":1.4,"stroke-dasharray":"5 4"}));
    const gW=iw/groups.length,sW=64;
    groups.forEach((g,gi)=>{const cx=m.l+gW*gi+gW/2,k=g.strips.length;
      g.strips.forEach((s,si)=>{const x=cx+(si-(k-1)/2)*(sW+16);
        s.vals.forEach((v,q)=>{const jit=((q*37)%11-5)*3.4; if(v<CL){const t=el("text",{x:x+jit,y:m.t+ih-3,"text-anchor":"middle",class:"val"});t.setAttribute("fill",s.col);t.textContent="▽";svg.appendChild(t);return;} svg.appendChild(el("circle",{cx:x+jit,cy:Y(v),r:3,fill:s.col,opacity:.55}));});
        const mv=med(s.vals); svg.appendChild(el("line",{x1:x-sW/2,x2:x+sW/2,y1:Y(mv),y2:Y(mv),stroke:s.col,"stroke-width":3}));
        const lab=el("text",{x,y:m.t+12,"text-anchor":"middle",class:"val"});lab.setAttribute("fill",s.col);
        lab.textContent=s.col===C.alone?"median "+BM.toFixed(4):`${(mv-BM)/BM*100>0?"+":""}${((mv-BM)/BM*100).toFixed(1)}%`;svg.appendChild(lab);
        const nn=el("text",{x,y:m.t+ih+40,"text-anchor":"middle",class:"axis"});nn.textContent=`n = ${s.vals.length}`;svg.appendChild(nn);});
      const tl=el("text",{x:cx,y:m.t+ih+22,"text-anchor":"middle",class:"lab"});tl.setAttribute("fill",cInk);tl.textContent=g.label;svg.appendChild(tl);});
    svg.appendChild(el("line",{x1:m.l,x2:m.l+iw,y1:m.t+ih,y2:m.t+ih,stroke:cRule}));
    const yl=el("text",{x:m.l,y:m.t-20,class:"axis"});yl.textContent="macro-F1 of each run · dashed line = median of alone ("+BM.toFixed(4)+") · % = median vs alone median";svg.appendChild(yl);
    if(hidden.length){const hn=el("text",{x:m.l+iw,y:m.t+ih-26,"text-anchor":"end",class:"axis"});hn.textContent=`▽ = ${hidden.length} run${hidden.length>1?"s":""} below ${CL} (lowest ${Math.min(...hidden).toFixed(3)}), drawn at the axis`;svg.appendChild(hn);}
  })();}catch(e){console.error('figure failed',e);}

  // ---- material_type per-class P/R + confusion matrices (ported from the dedicated page) ----
  try{(function(){
    const prf=M=>CLASSES.map((c,i)=>{const tp=M[i][i],sup=M[i].reduce((a,b)=>a+b,0),pred=M.reduce((a,r)=>a+r[i],0);return{cls:c,recall:sup?tp/sup:0,precision:pred?tp/pred:0};});
    const PRF={single:prf(CM.single.m),multi:prf(CM.multi.m)};
    const svg=document.getElementById("fig-mt-prf"); if(svg){svg.textContent="";
      const W=900,H=300,m={t:30,r:18,b:44,l:66}; const iw=W-m.l-m.r,ih=H-m.t-m.b; svg.setAttribute("width",W);svg.setAttribute("height",H); const panelW=iw/2-22;
      [["recall",0,"Recall — unchanged"],["precision",panelW+44,"Precision — roughly doubles"]].forEach(([metric,ox,title])=>{const x0=m.l+ox; const Y=v=>m.t+ih-v*ih;
        for(let v=0;v<=1.001;v+=.25){svg.appendChild(el("line",{x1:x0,x2:x0+panelW,y1:Y(v),y2:Y(v),stroke:cSoft}));if(ox===0){const tx=el("text",{x:x0-10,y:Y(v)+4,"text-anchor":"end",class:"axis"});tx.textContent=(v*100).toFixed(0)+"%";svg.appendChild(tx);}}
        const ttl=el("text",{x:x0,y:m.t-12,class:"lab"});ttl.setAttribute("fill",cInk);ttl.textContent=title;svg.appendChild(ttl);
        const bw=panelW/5,barW=Math.min(15,bw/3);
        CLASSES.forEach((c,i)=>{const cx=x0+bw*i+bw/2;[["single",C.xfer,-1],["multi",C.warm,1]].forEach(([arm,col,side])=>{const v=PRF[arm][i][metric];const bx=cx+side*3+(side<0?-barW:0);svg.appendChild(el("rect",{x:bx,y:Y(v),width:barW,height:Math.max(1,m.t+ih-Y(v)),fill:col,rx:2}));});
          const lab=el("text",{x:cx,y:m.t+ih+17,"text-anchor":"middle",class:"axis"});lab.textContent=c;svg.appendChild(lab);});
        svg.appendChild(el("line",{x1:x0,x2:x0+panelW,y1:m.t+ih,y2:m.t+ih,stroke:cRule}));});
      const note=el("text",{x:m.l,y:H-8,class:"axis"});note.textContent="test rows per run: DAC 1 · DQC 3 · IAC 24 · IQC 28 · others 7,298";svg.appendChild(note);}
    const cmv=document.getElementById("fig-mt-cm"); if(cmv){cmv.textContent="";
      const cell=52,W=980,H=410,m={t:76,l:104},gridW=cell*5,gap=110; cmv.setAttribute("width",W);cmv.setAttribute("height",H);
      [["single",0,"alone"],["multi",gridW+gap,"xfer (multi-task)"]].forEach(([arm,ox,title])=>{const M=CM[arm].m,runs=CM[arm].runs,x0=m.l+ox;
        const ttl=el("text",{x:x0,y:m.t-46,class:"lab"});ttl.setAttribute("fill",cInk);ttl.textContent=title;cmv.appendChild(ttl);
        const sub=el("text",{x:x0,y:m.t-31,class:"axis"});sub.textContent=runs+" runs, summed";cmv.appendChild(sub);
        const pl=el("text",{x:x0+gridW,y:m.t-31,"text-anchor":"end",class:"axis"});pl.textContent="PREDICTED \u2192";cmv.appendChild(pl);
        CLASSES.forEach((c,j)=>{const th=el("text",{x:x0+cell*j+cell/2,y:m.t-11,"text-anchor":"middle",class:"axis"});th.setAttribute("fill",cInk);th.textContent=c==="others"?"other":c;cmv.appendChild(th);});
        if(ox===0){const tl=el("text",{x:x0-78,y:m.t+cell*2.5,"text-anchor":"middle",class:"axis"});tl.setAttribute("transform",`rotate(-90 ${x0-78} ${m.t+cell*2.5})`);tl.textContent="TRUE CLASS";cmv.appendChild(tl);}
        M.forEach((row,i)=>{const tot=row.reduce((a,b)=>a+b,0),isO=i===4;
          if(ox===0){const rl=el("text",{x:x0-12,y:m.t+cell*i+cell/2+4,"text-anchor":"end",class:"axis"});rl.setAttribute("fill",cInk);rl.textContent=CLASSES[i];cmv.appendChild(rl);}
          if(isO)cmv.appendChild(el("rect",{x:x0-4,y:m.t+cell*i-4,width:gridW+6,height:cell+2,rx:3,fill:"none",stroke:C.xfer,"stroke-width":1.4,"stroke-dasharray":"3 3",opacity:.85}));
          row.forEach((v,j)=>{const f=tot?v/tot:0,diag=i===j;const alpha=f>0?0.06+0.34*Math.min(1,Math.sqrt(f*4)):0;
            cmv.appendChild(el("rect",{x:x0+cell*j+1,y:m.t+cell*i+1,width:cell-2,height:cell-2,rx:2,fill:diag?C.warm:C.xfer,opacity:alpha}));
            if(diag)cmv.appendChild(el("rect",{x:x0+cell*j+1.5,y:m.t+cell*i+1.5,width:cell-3,height:cell-3,rx:2,fill:"none",stroke:C.warm,"stroke-width":1.4}));
            if(v===0)return;const pct=f>=.1?(f*100).toFixed(0):f>=.01?(f*100).toFixed(1):(f*100).toFixed(2);
            const tv=el("text",{x:x0+cell*j+cell/2,y:m.t+cell*i+cell/2+1,"text-anchor":"middle",class:"axis"});tv.setAttribute("fill",cInk);tv.setAttribute("font-size","12.5");tv.textContent=pct;cmv.appendChild(tv);
            const cv=el("text",{x:x0+cell*j+cell/2,y:m.t+cell*i+cell/2+13,"text-anchor":"middle",class:"axis"});cv.setAttribute("font-size","10.5");cv.setAttribute("opacity",".8");cv.textContent=(v/runs)>=100?Math.round(v/runs).toLocaleString():(v/runs).toFixed(v/runs<10?1:0);cmv.appendChild(cv);});
          const sup=el("text",{x:x0+gridW+10,y:m.t+cell*i+cell/2+4,class:"axis"});sup.setAttribute("font-size","12");sup.textContent=(tot/runs)>=100?Math.round(tot/runs).toLocaleString():(tot/runs).toFixed(tot/runs<10?1:0);cmv.appendChild(sup);});
        const sh=el("text",{x:x0+gridW+10,y:m.t-11,class:"axis"});sh.setAttribute("font-size","11");sh.textContent="rows";cmv.appendChild(sh);const sh2=el("text",{x:x0+gridW+10,y:m.t-1,class:"axis"});sh2.setAttribute("font-size","11");sh2.textContent="/run";cmv.appendChild(sh2);
        const leak=M[4].slice(0,4).reduce((a,b)=>a+b,0);const lk=el("text",{x:x0,y:m.t+cell*5+26,class:"val"});lk.setAttribute("fill",C.xfer);lk.setAttribute("font-size","13.5");lk.textContent="others misfiled as rare: "+(leak/M[4].reduce((a,b)=>a+b,0)*100).toFixed(2)+"%";cmv.appendChild(lk);
        const lk2=el("text",{x:x0,y:m.t+cell*5+42,class:"axis"});lk2.textContent=Math.round(leak/runs)+" rows per run";cmv.appendChild(lk2);});
      }
  })();}catch(e){console.error('figure failed',e);}

  // ---- extensive / intensive / classification ----
  try{(function(){
    const svg=document.getElementById("fig-kind"); if(!svg) return; svg.textContent="";
    const rows=KIND.slice().sort((a,b)=>a[3]-b[3]); const W=960,rowH=26,m={t:30,r:40,b:44,l:190}; const H=m.t+rows.length*rowH+m.b; svg.setAttribute("height",H); const iw=W-m.l-m.r;
    const lo=-12,hi=24; const X=v=>m.l+(Math.max(lo,Math.min(hi,v))-lo)/(hi-lo)*iw; const col={extensive:C.xfer,intensive:C.frz,classification:C.warm};
    for(let v=-12;v<=24.01;v+=4){svg.appendChild(el("line",{x1:X(v),x2:X(v),y1:m.t-8,y2:m.t+rows.length*rowH,stroke:v===0?cRule:cSoft,"stroke-width":v===0?1.6:1}));const tx=el("text",{x:X(v),y:m.t-14,"text-anchor":"middle",class:"axis"});tx.textContent=(v>0?"+":"")+v+"%";svg.appendChild(tx);}
    rows.forEach(([task,kind,n,rel,matters,sep,se],i)=>{const y=m.t+i*rowH+rowH/2,c=col[kind],base=null;
      const nm=el("text",{x:m.l-12,y:y+4,"text-anchor":"end",class:"name"});nm.setAttribute("fill",kind==="extensive"?c:cInk);nm.textContent=pretty(task);svg.appendChild(nm);
      svg.appendChild(el("circle",{cx:X(rel),cy:y,r:5.5,fill:matters?c:cSurf,stroke:c,"stroke-width":1.8}));});
    const ax=el("text",{x:m.l+iw/2,y:H-10,"text-anchor":"middle",class:"axis"});ax.textContent="warm-start change vs training alone (%)";svg.appendChild(ax);
  })();}catch(e){console.error('figure failed',e);}

  // ---- representative strips: every run as a dot, the median as a bar, three-line labels ----
  try{document.querySelectorAll("svg.repfig").forEach(svg=>{
    svg.textContent=""; const task=svg.dataset.task; const d=REP[task];
    const W=960,H=300,m={t:34,r:20,b:86,l:70}; const iw=W-m.l-m.r,ih=H-m.t-m.b; svg.setAttribute("width",W);svg.setAttribute("height",H);
    const arms=[["alone",d.single,C.alone,"single","seeds"],["xfer",d.xfer,C.xfer,"xfer","runs"],["frozen",d.ftz,C.frz,"ftz","runs"],["warm-start",d.ftf,C.warm,"ftf","runs"]];
    const all=arms.flatMap(a=>a[1]); let lo=Math.min(...all),hi=Math.max(...all); const pad=(hi-lo)*0.15||0.02; lo-=pad;hi+=pad;
    const Y=v=>m.t+ih-(v-lo)/(hi-lo)*ih; const step=(hi-lo)/4;
    for(let k=0;k<=4;k++){const v=lo+k*step;svg.appendChild(el("line",{x1:m.l,x2:m.l+iw,y1:Y(v),y2:Y(v),stroke:cSoft}));const t=el("text",{x:m.l-10,y:Y(v)+4,"text-anchor":"end",class:"axis"});t.textContent=v.toFixed(3);svg.appendChild(t);}
    const am=med(d.single); svg.appendChild(el("line",{x1:m.l,x2:m.l+iw,y1:Y(am),y2:Y(am),stroke:C.alone,"stroke-width":1.4,"stroke-dasharray":"5 4"}));
    const colW=iw/arms.length;
    arms.forEach(([name,vals,col,key,unit],i)=>{
      const cx=m.l+colW*i+colW/2;
      vals.forEach((v,q)=>{const jit=((q*37)%11-5)*2.4; svg.appendChild(el("circle",{cx:cx+jit,cy:Y(v),r:5,fill:col,opacity:.75}));});
      const mv=med(vals); svg.appendChild(el("line",{x1:cx-30,x2:cx+30,y1:Y(mv),y2:Y(mv),stroke:col,"stroke-width":3}));
      const lab=el("text",{x:cx,y:m.t+ih+22,"text-anchor":"middle",class:"lab"}); lab.setAttribute("fill",col); lab.textContent=name; svg.appendChild(lab);
      const lines=[`n = ${vals.length} ${unit}`,`median ${mv.toFixed(4)}`]; const ep=d.epochs&&d.epochs[key]; if(ep!=null) lines.push(`${ep} epochs (median)`);
      lines.forEach((s,q)=>{const t=el("text",{x:cx,y:m.t+ih+40+q*16,"text-anchor":"middle",class:"axis"}); t.textContent=s; svg.appendChild(t);});
    });
    svg.appendChild(el("line",{x1:m.l,x2:m.l+iw,y1:m.t+ih,y2:m.t+ih,stroke:cRule}));
    const ttl=el("text",{x:m.l,y:m.t-14,class:"axis"}); ttl.textContent=`${pretty(task)} · ${task==="material_type"?"macro-F1":"R²"} of each run · dashed line = median of alone`; svg.appendChild(ttl);
  });}catch(e){console.error('repfig failed',e);}
}
draw();
// scale every figure to its container: fixed pixel widths become a viewBox, so nothing is clipped on a narrow viewport
document.querySelectorAll("svg").forEach(s=>{const w=+s.getAttribute("width"),h=+s.getAttribute("height");if(w&&h&&!s.getAttribute("viewBox")){s.setAttribute("viewBox",`0 0 ${w} ${h}`);s.style.width="100%";s.style.maxWidth=Math.round(w*1.4)+"px";s.style.height="auto";}});
const mq=window.matchMedia("(prefers-color-scheme: dark)"); mq.addEventListener&&mq.addEventListener("change",draw);
new MutationObserver(draw).observe(document.documentElement,{attributes:true,attributeFilter:["data-theme"]});
})();
