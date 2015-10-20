#!/usr/bin/env python3

from useful.small import OrderedDefaultDict
from useful.mstring import prints
from useful.mystruct import DefaultStruct
from collections import OrderedDict
from scipy.stats.stats import pearsonr
import numpy as np
from itertools import combinations, permutations

##############
# AMD Q'n'C  #
##############

results = {
  'fx_qnc': {
    # ./perforator.py -t "func=isolated_performance,interval=180000,warmup=15" -b
    # 27m
    'isolated': {'sdagp': 0.3948170590666076, 'matrix': 1.3258581954383786, 'blosc': 0.9027438996510717, 'sdag': 0.9878602828997607, 'pgbench': 0.41685737557876734, 'wordpress': 0.7410608402361379, 'static': 0.8154100675488877, 'ffmpeg': 1.3513974442660572},

    # time ./perforator.py -t "func=interference,mode=sibling,interval=180000,warmup=15" -d -b
    # ~ 2H
    'joint': {
      ('sdagp', 'wordpress'): [0.3430965814443583, 0.6365587555884425],
      ('ffmpeg', 'wordpress'): [1.0371735256331283, 0.6172619689305328],
      ('matrix', 'wordpress'): [1.1623467669678447, 0.5639848862800257],
      ('pgbench', 'pgbench'): [0.28722618220913865, 0.28709351888619156],
      ('blosc', 'blosc'): [0.7025859812564249, 0.7027715000892913],
      ('ffmpeg', 'sdag'): [1.080037286291761, 0.7977716815055407],
      ('sdagp', 'static'): [0.33730711593981755, 0.6854684429849044],
      ('ffmpeg', 'pgbench'): [0.9891276415528788, 0.3226154637323915],
      ('sdag', 'wordpress'): [0.7928237256451539, 0.6475012352180136],
      ('blosc', 'sdagp'): [0.7824750410519876, 0.3241118332968762],
      ('pgbench', 'sdag'): [0.33480011720968217, 0.7579838913352915],
      ('sdag', 'static'): [0.7234628348873473, 0.7095840479048232],
      ('matrix', 'static'): [0.9353918747373895, 0.6196772582470589],
      ('blosc', 'sdag'): [0.7842055974955036, 0.7700028477168415],
      ('blosc', 'ffmpeg'): [0.7693863497446234, 1.043561520431615],
      ('ffmpeg', 'static'): [0.9710992869848395, 0.6868290179195146],
      ('ffmpeg', 'matrix'): [0.9754557463812821, 1.1227619197438612],
      ('matrix', 'sdag'): [1.1932720727886252, 0.7247413614793661],
      ('pgbench', 'wordpress'): [0.3142125959315135, 0.5697829822839143],
      ('ffmpeg', 'ffmpeg'): [1.0146541626178265, 1.0127402742621707],
      ('static', 'static'): [0.6037023380310536, 0.6059869746359225],
      ('pgbench', 'static'): [0.27572877510582033, 0.6584913054632456],
      ('pgbench', 'sdagp'): [0.3083721206947051, 0.3764763257495822],
      ('blosc', 'wordpress'): [0.7798164094890779, 0.6085187872840804],
      ('matrix', 'pgbench'): [1.1117117304701223, 0.27317619197466153],
      ('wordpress', 'wordpress'): [0.5925345116240807, 0.5883107667267231],
      ('blosc', 'static'): [0.6954315879927159, 0.6480019596008798],
      ('static', 'wordpress'): [0.6773638363494453, 0.5606111166562152],
      ('sdag', 'sdagp'): [0.7705793608170706, 0.3419872210713658],
      ('blosc', 'pgbench'): [0.7660014096432362, 0.2969626005969703],
      ('matrix', 'matrix'): [0.8348071133815028, 0.8353572578683788],
      ('matrix', 'sdagp'): [1.1388142783070707, 0.269629343036394],
      ('sdagp', 'sdagp'): [0.3276833203924927, 0.34010497284277147],
      ('blosc', 'matrix'): [0.6823663368253011, 1.0420031109332408],
      ('sdag', 'sdag'): [0.765633349733418, 0.77455869164682],
      ('ffmpeg', 'sdagp'): [1.094010419099382, 0.41054127238945776]}
  },
  'fx_normale': {
    'ht_map': {0: [3], 1: [4], 2: [6], 3: [0], 4: [1], 5: [7], 6: [2], 7: [5]},
    # ./perforator.py -t "func=isolated_performance,interval=180000,warmup=15" -b
    # 27m
    'isolated': OrderedDict([
      ('blosc', 0.8956988716825668),
      ('ffmpeg', 1.3543917558196419),
      ('matrix', 1.3439633916695972),
      ('pgbench', 0.41916398218418577),
      ('sdag', 0.9968216451194699),
      ('sdagp', 0.41516906736083686),
      ('static', 0.8180926271425154),
      ('wordpress', 0.7958097549237776)]),
    # time ./perforator.py -t "func=interference,mode=sibling,interval=180000,warmup=15" -d -b
    # ~ 2H
    'joint_near': OrderedDict([
      (('blosc', 'blosc'), [0.6951163554816507, 0.6967882442039682]),
      (('blosc', 'ffmpeg'), [0.7655300298011922, 1.0488565060957318]),
      (('blosc', 'matrix'), [0.6804293427438444, 1.0420873037674618]),
      (('blosc', 'pgbench'), [0.7676326255775389, 0.2970924409518037]),
      (('blosc', 'sdag'), [0.792557450625611, 0.7727798077420506]),
      (('blosc', 'sdagp'), [0.7813613821774313, 0.31141146154586313]),
      (('blosc', 'static'), [0.692311104412252, 0.6501924560555712]),
      (('blosc', 'wordpress'), [0.7825297621610223, 0.6078305025903767]),
      (('ffmpeg', 'ffmpeg'), [1.017937960281586, 1.0194526718864398]),
      (('ffmpeg', 'matrix'), [0.9763584753510288, 1.125765714687391]),
      (('ffmpeg', 'pgbench'), [0.9904728032413691, 0.3217304174285826]),
      (('ffmpeg', 'sdag'), [1.0780542320161544, 0.79981943976313]),
      (('ffmpeg', 'sdagp'), [1.1224257617566458, 0.34447258959881316]),
      (('ffmpeg', 'static'), [0.9697644012188444, 0.686734939447787]),
      (('ffmpeg', 'wordpress'), [1.031761605669347, 0.6113607674070324]),
      (('matrix', 'matrix'), [0.8752844426499016, 0.873583247019585]),
      (('matrix', 'pgbench'), [1.109597044350408, 0.2750654177691763]),
      (('matrix', 'sdag'), [1.1854905137372744, 0.7242317017598909]),
      (('matrix', 'sdagp'), [1.1306420359440963, 0.2704450007581889]),
      (('matrix', 'static'), [0.932254871691537, 0.6219900635914979]),
      (('matrix', 'wordpress'), [1.1667507618793012, 0.5638546270943112]),
      (('pgbench', 'pgbench'), [0.28592611032670057, 0.2870339915350473]),
      (('pgbench', 'sdag'), [0.3342125852697608, 0.7575299183364784]),
      (('pgbench', 'sdagp'), [0.31676436481391784, 0.326699399459858]),
      (('pgbench', 'static'), [0.2740185708407565, 0.6577251643836423]),
      (('pgbench', 'wordpress'), [0.3136434211965589, 0.569500037093543]),
      (('sdag', 'sdag'), [0.775771838689932, 0.777307237713907]),
      (('sdag', 'sdagp'), [0.7691081972218102, 0.3817947438265611]),
      (('sdag', 'static'), [0.7251072644170558, 0.71305627288238]),
      (('sdag', 'wordpress'), [0.7835466102851865, 0.6366716470217719]),
      (('sdagp', 'sdagp'), [0.3992444221999048, 0.3105208172023552]),
      (('sdagp', 'static'), [0.28230852498287573, 0.6937938178478076]),
      (('sdagp', 'wordpress'), [0.38188513559754406, 0.6271128125816242]),
      (('static', 'static'), [0.6035574748561862, 0.6085196305043862]),
      (('static', 'wordpress'), [0.6795300714512448, 0.5609176751209078]),
      (('wordpress', 'wordpress'), [0.5975575068237051, 0.590820644167468])]),
    'isolated_stats':
    # ./perforator.py -t "func=all_events,interval=900000,warmup=15" -b
    {'sdagp': {'cpu-clock': 899149, 'cycles': 2769271032062, 'stalled-cycles-backend': 1835715187128, 'stalled-cycles-frontend': 254297119578, 'L1-dcache-prefetches': 38741148856, 'dTLB-load-misses': 24611240820, 'branch-instructions': 283610238344, 'page-faults': 274, 'LLC-load-misses': 12797164696, 'context-switches': 69906, 'branch-load-misses': 3664676752, 'alignment-faults': 0, 'L1-dcache-load-misses': 55536350217, 'branch-loads': 283268643165, 'LLC-stores': 27747122349, 'iTLB-load-misses': 2402497, 'instructions': 1143886389899, 'emulation-faults': 0, 'minor-faults': 273, 'dTLB-loads': 486437360278, 'L1-dcache-prefetch-misses': 0, 'dummy': 0, 'cache-references': 410643928233, 'task-clock': 899146, 'L1-dcache-loads': 487251004990, 'L1-icache-loads': 411669056919, 'L1-icache-load-misses': 2738333499, 'L1-dcache-stores': 66908822690, 'cpu-migrations': 1, 'LLC-loads': 69934987259, 'branch-misses': 3674711534, 'L1-icache-prefetches': 29853001, 'major-faults': 0, 'iTLB-loads': 411158473884, 'cache-misses': 2731332372},
    'blosc': {'cpu-clock': 899057, 'cycles': 2770705227214, 'stalled-cycles-backend': 1203235831051, 'stalled-cycles-frontend': 284590377033, 'L1-dcache-prefetches': 42718280897, 'dTLB-load-misses': 29338685, 'branch-instructions': 441406574268, 'page-faults': 61, 'LLC-load-misses': 16138580024, 'context-switches': 67699, 'branch-load-misses': 12420055470, 'alignment-faults': 0, 'L1-dcache-load-misses': 43013090999, 'branch-loads': 440293069623, 'LLC-stores': 33584984234, 'iTLB-load-misses': 4599382, 'instructions': 2477599729770, 'emulation-faults': 0, 'minor-faults': 61, 'dTLB-loads': 912238501748, 'L1-dcache-prefetch-misses': 0, 'dummy': 0, 'cache-references': 814698017841, 'task-clock': 899050, 'L1-dcache-loads': 910207818269, 'L1-icache-loads': 814036675223, 'L1-icache-load-misses': 394879836, 'L1-dcache-stores': 77716918323, 'cpu-migrations': 2, 'LLC-loads': 78173397663, 'branch-misses': 12403459894, 'L1-icache-prefetches': 8195795, 'major-faults': 0, 'iTLB-loads': 814072170432, 'cache-misses': 394079072},
    'ffmpeg': {'cpu-clock': 899142, 'cycles': 2757636781715, 'stalled-cycles-backend': 971114222232, 'stalled-cycles-frontend': 323146299979, 'L1-dcache-prefetches': 43482803917, 'dTLB-load-misses': 29001381, 'branch-instructions': 216148490785, 'page-faults': 143, 'LLC-load-misses': 2870859587, 'context-switches': 65798, 'branch-load-misses': 13101732168, 'alignment-faults': 0, 'L1-dcache-load-misses': 62984307931, 'branch-loads': 216167993679, 'LLC-stores': 5356853974, 'iTLB-load-misses': 5053090, 'instructions': 3741420907229, 'emulation-faults': 0, 'minor-faults': 143, 'dTLB-loads': 1381263245410, 'L1-dcache-prefetch-misses': 0, 'dummy': 0, 'cache-references': 1023811632506, 'task-clock': 899140, 'L1-dcache-loads': 1384791406819, 'L1-icache-loads': 1030788191583, 'L1-icache-load-misses': 24312817516, 'L1-dcache-stores': 62804014072, 'cpu-migrations': 0, 'LLC-loads': 89406234413, 'branch-misses': 13094860762, 'L1-icache-prefetches': 2697279089, 'major-faults': 0, 'iTLB-loads': 1030454932489, 'cache-misses': 24287734385},
    'static': {'cpu-clock': 898975, 'cycles': 2765233533936, 'stalled-cycles-backend': 1294974903809, 'stalled-cycles-frontend': 380895176064, 'L1-dcache-prefetches': 96337265389, 'dTLB-load-misses': 246192291, 'branch-instructions': 225205852608, 'page-faults': 0, 'LLC-load-misses': 13453265902, 'context-switches': 68587, 'branch-load-misses': 3088659356, 'alignment-faults': 0, 'L1-dcache-load-misses': 46045644313, 'branch-loads': 225196990034, 'LLC-stores': 37661743553, 'iTLB-load-misses': 28555390, 'instructions': 2247614317353, 'emulation-faults': 0, 'minor-faults': 0, 'dTLB-loads': 1115577620314, 'L1-dcache-prefetch-misses': 0, 'dummy': 0, 'cache-references': 490884261896, 'task-clock': 898963, 'L1-dcache-loads': 1117962955598, 'L1-icache-loads': 490784280791, 'L1-icache-load-misses': 5706997951, 'L1-dcache-stores': 89270266076, 'cpu-migrations': 2, 'LLC-loads': 96754357044, 'branch-misses': 3093177403, 'L1-icache-prefetches': 177130118, 'major-faults': 0, 'iTLB-loads': 490747586720, 'cache-misses': 5688189002},
    'wordpress': {'cpu-clock': 897622, 'cycles': 2759073186251, 'stalled-cycles-backend': 690421560572, 'stalled-cycles-frontend': 551197002391, 'L1-dcache-prefetches': 50014560436, 'dTLB-load-misses': 470823204, 'branch-instructions': 427692525185, 'page-faults': 0, 'LLC-load-misses': 3605261984, 'context-switches': 118785, 'branch-load-misses': 24530556749, 'alignment-faults':0, 'L1-dcache-load-misses': 40911737726, 'branch-loads': 427085893076, 'LLC-stores': 7685774240, 'iTLB-load-misses': 162451693, 'instructions': 2158268755085, 'emulation-faults': 0, 'minor-faults': 0, 'dTLB-loads': 1217050918460, 'L1-dcache-prefetch-misses': 0, 'dummy': 0, 'cache-references': 1240126225846, 'task-clock': 897619, 'L1-dcache-loads': 1216273543967, 'L1-icache-loads': 1240650544228, 'L1-icache-load-misses': 27563517581, 'L1-dcache-stores': 42997210535, 'cpu-migrations': 0, 'LLC-loads': 78504648664, 'branch-misses': 24517322418, 'L1-icache-prefetches': 51433214, 'major-faults': 0, 'iTLB-loads': 1241003901861, 'cache-misses': 27536602675},
    'matrix': {'cpu-clock': 899036, 'cycles': 2765306965058, 'stalled-cycles-backend': 68459542971, 'stalled-cycles-frontend': 964561518809, 'L1-dcache-prefetches': 110062452114, 'dTLB-load-misses': 14846624, 'branch-instructions': 463441681833, 'page-faults': 85, 'LLC-load-misses': 2857598992, 'context-switches': 65156, 'branch-load-misses': 249805333, 'alignment-faults': 0, 'L1-dcache-load-misses': 9757339630, 'branch-loads': 463232977822, 'LLC-stores': 58242924301, 'iTLB-load-misses': 1260775, 'instructions': 3702554265256, 'emulation-faults': 0,'minor-faults': 85, 'dTLB-loads': 1424143743592, 'L1-dcache-prefetch-misses': 0, 'dummy': 0, 'cache-references': 957437892971, 'task-clock': 899030, 'L1-dcache-loads': 1423682647002, 'L1-icache-loads': 957464282788, 'L1-icache-load-misses': 133716322, 'L1-dcache-stores': 114959360213, 'cpu-migrations': 2, 'LLC-loads': 115122619237, 'branch-misses': 249989560, 'L1-icache-prefetches': 261345, 'major-faults': 0, 'iTLB-loads': 957346014488, 'cache-misses': 133301515},
    'sdag': {'cpu-clock': 899183, 'cycles': 2776336387761, 'stalled-cycles-backend': 1004912857772, 'stalled-cycles-frontend': 230436289133, 'L1-dcache-prefetches': 86268504151, 'dTLB-load-misses': 117703509, 'branch-instructions': 611459601526, 'page-faults': 110, 'LLC-load-misses': 787395116, 'context-switches': 65391, 'branch-load-misses': 13025296518, 'alignment-faults': 0, 'L1-dcache-load-misses': 58904567815, 'branch-loads': 611465765221, 'LLC-stores': 3140770905, 'iTLB-load-misses': 3762514, 'instructions': 2758172786455, 'emulation-faults': 0, 'minor-faults': 110, 'dTLB-loads': 1299410428973, 'L1-dcache-prefetch-misses': 0, 'dummy': 0, 'cache-references': 1000853712400, 'task-clock': 899181, 'L1-dcache-loads': 1304167765653, 'L1-icache-loads': 1000174042423, 'L1-icache-load-misses': 8477022554, 'L1-dcache-stores': 65975213205, 'cpu-migrations': 0, 'LLC-loads': 75696023982, 'branch-misses': 13002383995, 'L1-icache-prefetches': 17841869, 'major-faults': 0, 'iTLB-loads': 1001021511302, 'cache-misses': 8481466422},
    'pgbench': {'cpu-clock': 872559, 'cycles': 2222713292020, 'stalled-cycles-backend': 276470015346, 'stalled-cycles-frontend': 1090728319832, 'L1-dcache-prefetches': 24554959043, 'dTLB-load-misses': 2110885875, 'branch-instructions': 190912012087, 'page-faults': 25, 'LLC-load-misses': 5436584103, 'context-switches': 5278924, 'branch-load-misses': 16718543288, 'alignment-faults': 0, 'L1-dcache-load-misses': 23872995271, 'branch-loads': 190877065711, 'LLC-stores': 10987791687, 'iTLB-load-misses': 2152068318, 'instructions': 925972757585, 'emulation-faults': 0, 'minor-faults': 25, 'dTLB-loads': 543309393550, 'L1-dcache-prefetch-misses': 0, 'dummy': 0, 'cache-references': 839128963692, 'task-clock': 872598, 'L1-dcache-loads': 542480765530, 'L1-icache-loads': 840275039855, 'L1-icache-load-misses': 51491000326, 'L1-dcache-stores': 25434785636, 'cpu-migrations': 2, 'LLC-loads': 82171334264, 'branch-misses': 16711784728, 'L1-icache-prefetches': 94799408, 'major-faults': 0, 'iTLB-loads': 836597669547, 'cache-misses': 51567238607}},

  # TODO: quick and dirty
    'joint_far': OrderedDict([
      (('blosc', 'blosc'), [0.8791980020581556, 0.8835425143956932]),
      (('blosc', 'ffmpeg'), [0.886446986939226, 1.3300011850203686]),
      (('blosc', 'matrix'), [0.8404601354137381, 1.2203116772889326]),
      (('blosc', 'pgbench'), [0.8927169051956728, 0.39732802052095856]),
      (('blosc', 'sdag'), [0.8982940724828407, 0.9666228159589145]),
      (('blosc', 'sdagp'), [0.8745958993390688, 0.3862550389137518]),
      (('blosc', 'static'), [0.8712692610165333, 0.7861864262266686]),
      (('blosc', 'wordpress'), [0.8981368041376337, 0.7743616672600262]),
      (('ffmpeg', 'ffmpeg'), [1.3393011665174903, 1.346307787940447]),
      (('ffmpeg', 'matrix'), [1.2707696153045267, 1.2891815197872145]),
      (('ffmpeg', 'pgbench'), [1.345934075924994, 0.4146136280766169]),
      (('ffmpeg', 'sdag'), [1.353415129300496, 0.9872681004403706]),
      (('ffmpeg', 'sdagp'), [1.34344397423967, 0.3890065001676034]),
      (('ffmpeg', 'static'), [1.3324166151693675, 0.8098712833730397]),
      (('ffmpeg', 'wordpress'), [1.340570441298122, 0.7843218856749316]),
      (('matrix', 'matrix'), [1.035358703275237, 1.0314307192651773]),
      (('matrix', 'pgbench'), [1.2798569226342198, 0.37993072434561487]),
      (('matrix', 'sdag'), [1.2927058559703777, 0.9360810038578065]),
      (('matrix', 'sdagp'), [1.2609169503665953, 0.3709891102782426]),
      (('matrix', 'static'), [1.1543545340049632, 0.7108237252382593]),
      (('matrix', 'wordpress'), [1.2684453727892517, 0.7309928823900748]),
      (('pgbench', 'pgbench'), [0.4088520221625777, 0.40589566824552875]),
      (('pgbench', 'sdag'), [0.41072143790548327, 0.9897388820617588]),
      (('pgbench', 'sdagp'), [0.40180547923783616, 0.3870494046841379]),
      (('pgbench', 'static'), [0.3839856062052388, 0.8005951853955942]),
      (('pgbench', 'wordpress'), [0.4025814053450829, 0.7751203252948216]),
      (('sdag', 'sdag'), [1.0055142631984322, 0.97793464400214]),
      (('sdag', 'sdagp'), [0.9911738466366161, 0.42218898190189486]),
      (('sdag', 'static'), [0.9722299153073064, 0.81499649365868]),
      (('sdag', 'wordpress'), [0.9940407091850607, 0.7823669399855767]),
      (('sdagp', 'sdagp'), [0.42801817285802585, 0.38520405717840295]),
      (('sdagp', 'static'), [0.41306145405844513, 0.7986046456001192]),
      (('sdagp', 'wordpress'), [0.39190158642031575, 0.7735250062905058]),
      (('static', 'static'), [0.7753331284936542, 0.7754022609496972]),
      (('static', 'wordpress'), [0.8090521503919693, 0.7544492984033327]),
      (('wordpress', 'wordpress'), [0.7835413916098692, 0.7737217232572693])]),
  },
}


def print_sens_brut(ret):
  sensitivity = ret.sensitivity
  brutality = ret.brutality

  prints("\nBMARK       sensitivity    brutality")
  for bmark,degr in sorted(sensitivity.items(), key=lambda x: x[1], reverse=True):
    brut = brutality[bmark]
    prints("{bmark:<11} {degr:>8.2f} {brut:>10.2f}")


def print_correlation(ret):
  for event, (corr, p) in ret.correlation:
    if p > 0.05:
      continue
    prints("{event:<25} {corr:>8.4f} {p:>8.4f}")


def ratio(sh, iso, param):
  """ How rate of an event changes om shared env compared to isolated. """
  sh_param = sh[param] / sh['cycles']
  iso_param = iso[param] / iso['cycles']
  return sh_param / iso_param


def analyze(isolated, joint, stats=None):
  bmarks = sorted(isolated.keys())
  events = sorted(stats['blosc'])        # all events
  sensitivity = OrderedDefaultDict(int)
  brutality = OrderedDefaultDict(int)
  ret = DefaultStruct()

  # print header
  print()
  print("          ", end='')
  [prints("{bmark:>10}", end='') for bmark in bmarks]
  print()
  # print table
  for fg in bmarks:
    prints("{fg:<10}", end='')
    for bg in bmarks:
      if (fg, bg) in joint:
        fg_perf, bg_perf = joint[fg, bg]
      else:
        bg_perf, fg_perf = joint[bg, fg]
      fg_degr = 1 - fg_perf / isolated[fg]
      bg_degr = 1 - bg_perf / isolated[bg]
      sensitivity[fg] += fg_degr
      sensitivity[bg] += bg_degr
      brutality[fg]   += bg_degr
      brutality[bg]   += fg_degr
      prints("{fg_degr:>10.1%}", end='')
    print()

  # normalize
  for k,v in sensitivity.items():
    sensitivity[k] = v / len(bmarks)
  for k,v in brutality.items():
    brutality[k] = v / len(bmarks)

  for themap, result in [(sensitivity, ret.sensitivity),
                         (brutality, ret.brutality)]:
    result.correlation = []
    result.regression  = {}

    for event in events:
      X, Y = [], []
      for bmark in bmarks:
        coeff = themap[bmark]
        count = stats[bmark][event]
        X.append(count)
        Y.append(coeff)
        # prints("{bmark:<10}: sens:{sens:<6.2f} brut:{brut:.2f}")
      corr, p = pearsonr(X, Y)
      if abs(p) > 0.05:
        continue
      result.correlation.append((event, corr, p))
      prints("correlation {event:<21} {corr:+.3f} {p:.3f}")
      result.regression[event] = np.polyfit(X, Y, 1)
    result.correlation.sort(key=lambda v: -abs(v[1]))
  return ret

  for (bmark1, bmark2), (ipc1_sh, ipc2_sh) in joint.items():
    ipc1_iso = isolated[bmark1]
    ipc2_iso = isolated[bmark2]
    ratio1   = ipc1_sh / ipc1_iso
    ratio2   = ipc2_sh / ipc2_iso
    prints("{bmark1:<10} {bmark2:<10}: {degr1:>.2f} {degr2:>8.2f}")


def predict(stats, correlation, regression):
  # prints("CORRR1: {correlation}")
  def _predict(task):
    # prints("CORRR2: {correlation}")
    event, corr, p = correlation[0]
    # prints("for {task} using {event} with correlation {corr:.3f}")
    a, b = regression[event]
    stat = stats[task]
    cnt  = stat[event]
    return a*cnt + b
  return _predict


def overhead(tasks, htmap, brutality_near, brutality_far):
  overhead = 0
  for task_cpu, task in enumerate(tasks):
    for other_cpu, other in enumerate(tasks):
      if task_cpu == other_cpu:
        continue
      if htmap[task_cpu] == [other_cpu]:
        overhead += brutality_near(task)
      else:
        overhead += brutality_far(task)
  return overhead


def estimate(alloc, htmap, sens_near, sens_far, brut_near, brut_far):
  degradation = 0
  for (a, cpu_a), (b, cpu_b) in combinations(alloc, 2):
    # print(a, b, cpu_a, cpu_b)
    if cpu_a in htmap[cpu_b]:
      brut, sens = brut_near, sens_near
    else:
      brut, sens = brut_far, sens_far

    degradation += sens(a) + sens(b) + brut(a) + brut(b)
  return degradation


def calc_regr():
  data = results['fx_normale']
  stats = data['isolated_stats']

  sibling = analyze(isolated=data['isolated'], joint=data['joint_near'], stats=stats)
  distant = analyze(isolated=data['isolated'], joint=data['joint_far'],  stats=stats)

  sens_near = predict(stats=stats,
                      correlation=sibling.sensitivity.correlation,
                      regression=sibling.sensitivity.regression)

  sens_far = predict(stats=stats,
                     correlation=distant.sensitivity.correlation,
                     regression=distant.sensitivity.regression)

  brut_near = predict(stats=stats,
                      correlation=sibling.brutality.correlation,
                      regression=sibling.brutality.regression)

  brut_far = predict(stats=stats,
                     correlation=distant.brutality.correlation,
                     regression=distant.brutality.regression)
  return sens_near, sens_far, brut_near, brut_far


def main():
  sens_near, sens_far, brut_near, brut_far = calc_regr()
  # brutality_far  = predict(stats=stats,
  #                          correlation=distant.correlation,
  #                          regression=distant.regression)

  # for bmark in stats:
  #   print("{:.3f}".format(brutality_near(bmark)))
  htmap = {0: [5], 1: [3], 2: [4], 3: [1], 4: [2], 5: [0], 6: [7], 7: [6]}
  # alloc = [("blosc",0), ("blosc",5)]
  # r = estimate(alloc, htmap=htmap, sens_near=sens_near, sens_far=sens_far, brut_near=brut_near, brut_far=brut_far)
  # tasks = [task for task, cpu in alloc]
  all_cpus = [0,1,2,3,4,5,6,7]
  tasks = ['sdagp', 'ffmpeg', 'pgbench', 'wordpress', 'matrix', 'matrix', 'blosc', 'wordpress']
  for cpus in permutations(all_cpus):
    print(cpus, tasks)
    r = estimate(zip(tasks, cpus), htmap=htmap, sens_near=sens_near, sens_far=sens_far, brut_near=brut_near, brut_far=brut_far)
    print(r, cpus)

if __name__ == '__main__':

  main()
