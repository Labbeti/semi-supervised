import augmentation_utils.spec_augmentations as spec_aug
import augmentation_utils.signal_augmentations as signal_aug

augmentations = dict(
    none="",
    # Spec augmentation
    flip_ud=spec_aug.VerticalFlip(1.0),
    flip_lr=spec_aug.HorizontalFlip(1.0),
    usn_1=spec_aug.UniformSignNoise(1.0, epsilon=0.001, mini=-80, maxi=0),
    usn_2=spec_aug.UniformSignNoise(1.0, epsilon=0.005, mini=-80, maxi=0),
    usn_3=spec_aug.UniformSignNoise(1.0, epsilon=0.010, mini=-80, maxi=0),
    usn_4=spec_aug.UniformSignNoise(1.0, epsilon=0.020, mini=-80, maxi=0),
    usn_5=spec_aug.UniformSignNoise(1.0, epsilon=0.030, mini=-80, maxi=0),
    usn_6=spec_aug.UniformSignNoise(1.0, epsilon=0.040, mini=-80, maxi=0),
    usn_7=spec_aug.UniformSignNoise(1.0, epsilon=0.050, mini=-80, maxi=0),
    usn_8=spec_aug.UniformSignNoise(1.0, epsilon=0.060, mini=-80, maxi=0),
    usn_9=spec_aug.UniformSignNoise(1.0, epsilon=0.070, mini=-80, maxi=0),
    usn_10=spec_aug.UniformSignNoise(1.0, epsilon=0.08, mini=-80, maxi=0),
    usn_11=spec_aug.UniformSignNoise(1.0, epsilon=0.09, mini=-80, maxi=0),
    usn_12=spec_aug.UniformSignNoise(1.0, epsilon=0.10, mini=-80, maxi=0),
    usn_13=spec_aug.UniformSignNoise(1.0, epsilon=0.20, mini=-80, maxi=0),
    usn_14=spec_aug.UniformSignNoise(1.0, epsilon=0.30, mini=-80, maxi=0),
    usn_15=spec_aug.UniformSignNoise(1.0, epsilon=0.40, mini=-80, maxi=0),
    usn_16=spec_aug.UniformSignNoise(1.0, epsilon=0.50, mini=-80, maxi=0),
    usn_17=spec_aug.UniformSignNoise(1.0, epsilon=0.60, mini=-80, maxi=0),
    usn_18=spec_aug.UniformSignNoise(1.0, epsilon=0.70, mini=-80, maxi=0),
    usn_19=spec_aug.UniformSignNoise(1.0, epsilon=0.80, mini=-80, maxi=0),
    usn_20=spec_aug.UniformSignNoise(1.0, epsilon=0.90, mini=-80, maxi=0),
    usn_21=spec_aug.UniformSignNoise(1.0, epsilon=1.00, mini=-80, maxi=0),
    usn_22=spec_aug.UniformSignNoise(1.0, epsilon=2.00, mini=-80, maxi=0),
    usn_23=spec_aug.UniformSignNoise(1.0, epsilon=3.00, mini=-80, maxi=0),
    usn_24=spec_aug.UniformSignNoise(1.0, epsilon=4.00, mini=-80, maxi=0),
    usn_25=spec_aug.UniformSignNoise(1.0, epsilon=5.00, mini=-80, maxi=0),
    usn_26=spec_aug.UniformSignNoise(1.0, epsilon=6.00, mini=-80, maxi=0),
    usn_27=spec_aug.UniformSignNoise(1.0, epsilon=7.00, mini=-80, maxi=0),
    usn_28=spec_aug.UniformSignNoise(1.0, epsilon=8.00, mini=-80, maxi=0),
    usn_29=spec_aug.UniformSignNoise(1.0, epsilon=9.00, mini=-80, maxi=0),
    n_10=spec_aug.Noise(1.0, 10),
    n_15=spec_aug.Noise(1.0, 15),
    n_20=spec_aug.Noise(1.0, 20),
    n_25=spec_aug.Noise(1.0, 25),
    n_30=spec_aug.Noise(1.0, 30),
    n_40=spec_aug.Noise(1.0, 40),
    n_10_05=spec_aug.Noise(0.5, 10),
    n_15_05=spec_aug.Noise(0.5, 15),
    n_20_05=spec_aug.Noise(0.5, 20),
    n_25_05=spec_aug.Noise(0.5, 25),
    n_30_05=spec_aug.Noise(0.5, 30),
    n_40_05=spec_aug.Noise(0.5, 40),
    fts_1=spec_aug.FractalTimeStretch(
        1.0, 0.1, rate=(0.8, 1.2), min_chunk_size=2, max_chunk_size=4
    ),
    fts_2=spec_aug.FractalTimeStretch(
        1.0, 0.2, rate=(0.8, 1.2), min_chunk_size=2, max_chunk_size=4
    ),
    fts_3=spec_aug.FractalTimeStretch(
        1.0, 0.3, rate=(0.8, 1.2), min_chunk_size=2, max_chunk_size=8
    ),
    fts_4=spec_aug.FractalTimeStretch(
        1.0, 0.4, rate=(0.8, 1.2), min_chunk_size=2, max_chunk_size=16
    ),
    fts_5=spec_aug.FractalTimeStretch(
        1.0, 0.5, rate=(0.8, 1.2), min_chunk_size=2, max_chunk_size=32
    ),
    fts_51=spec_aug.FractalTimeStretch(
        1.0, 0.5, rate=(0.8, 1.2), min_chunk_size=4, max_chunk_size=32
    ),
    fts_52=spec_aug.FractalTimeStretch(
        1.0, 0.5, rate=(0.8, 1.2), min_chunk_size=8, max_chunk_size=32
    ),
    fts_53=spec_aug.FractalTimeStretch(
        1.0, 0.5, rate=(0.8, 1.2), min_chunk_size=12, max_chunk_size=32
    ),
    fts_54=spec_aug.FractalTimeStretch(
        1.0, 0.5, rate=(0.8, 1.2), min_chunk_size=16, max_chunk_size=32
    ),
    fts_55=spec_aug.FractalTimeStretch(
        1.0, 0.5, rate=(0.8, 1.2), min_chunk_size=20, max_chunk_size=32
    ),
    ffs_1=spec_aug.FractalFreqStretch(
        1.0, 0.1, rate=(0.8, 1.2), min_chunk_size=2, max_chunk_size=4
    ),
    ffs_2=spec_aug.FractalFreqStretch(
        1.0, 0.2, rate=(0.8, 1.2), min_chunk_size=2, max_chunk_size=8
    ),
    ffs_3=spec_aug.FractalFreqStretch(
        1.0, 0.3, rate=(0.8, 1.2), min_chunk_size=2, max_chunk_size=12
    ),
    ffs_4=spec_aug.FractalFreqStretch(
        1.0, 0.4, rate=(0.8, 1.2), min_chunk_size=2, max_chunk_size=16
    ),
    ffs_5=spec_aug.FractalFreqStretch(
        1.0, 0.5, rate=(0.8, 1.2), min_chunk_size=2, max_chunk_size=20
    ),
    ffs_51=spec_aug.FractalFreqStretch(
        1.0, 0.5, rate=(0.8, 1.2), min_chunk_size=6, max_chunk_size=20
    ),
    ffs_52=spec_aug.FractalFreqStretch(
        1.0, 0.5, rate=(0.7, 1.3), min_chunk_size=6, max_chunk_size=20
    ),
    ffs_53=spec_aug.FractalFreqStretch(
        1.0, 0.5, rate=(0.6, 1.4), min_chunk_size=6, max_chunk_size=20
    ),
    ffs_54=spec_aug.FractalFreqStretch(
        1.0, 0.5, rate=(0.5, 1.5), min_chunk_size=6, max_chunk_size=20
    ),
    ffs_55=spec_aug.FractalFreqStretch(
        1.0, 0.5, rate=(0.4, 1.6), min_chunk_size=6, max_chunk_size=20
    ),
    ftd_1=spec_aug.FractalTimeDropout(1.0, 1, 2),
    ftd_2=spec_aug.FractalTimeDropout(1.0, 2, 4),
    ftd_3=spec_aug.FractalTimeDropout(1.0, 2, 8),
    ftd_4=spec_aug.FractalTimeDropout(1.0, 2, 16),
    ftd_5=spec_aug.FractalTimeDropout(1.0, 2, 32),
    ffd_1=spec_aug.FractalFrecDropout(1.0, 1, 2),
    ffd_2=spec_aug.FractalFrecDropout(1.0, 2, 4),
    ffd_3=spec_aug.FractalFrecDropout(1.0, 2, 8),
    ffd_4=spec_aug.FractalFrecDropout(1.0, 2, 10),
    ffd_5=spec_aug.FractalFrecDropout(1.0, 2, 12),
    rtd_1=spec_aug.RandomTimeDropout(1.0, dropout=0.01),
    rtd_2=spec_aug.RandomTimeDropout(1.0, dropout=0.05),
    rtd_3=spec_aug.RandomTimeDropout(1.0, dropout=0.1),
    rtd_4=spec_aug.RandomTimeDropout(1.0, dropout=0.2),
    rtd_5=spec_aug.RandomTimeDropout(1.0, dropout=0.3),
    rtd_6=spec_aug.RandomTimeDropout(1.0, dropout=0.4),
    rtd_7=spec_aug.RandomTimeDropout(1.0, dropout=0.5),
    rtd_8=spec_aug.RandomTimeDropout(1.0, dropout=0.6),
    rtd_9=spec_aug.RandomTimeDropout(1.0, dropout=0.7),
    rtd_10=spec_aug.RandomTimeDropout(1.0, dropout=0.8),
    rtd_11=spec_aug.RandomTimeDropout(1.0, dropout=0.9),
    rfd_1=spec_aug.RandomFreqDropout(1.0, dropout=0.01),
    rfd_2=spec_aug.RandomFreqDropout(1.0, dropout=0.05),
    rfd_3=spec_aug.RandomFreqDropout(1.0, dropout=0.1),
    rfd_4=spec_aug.RandomFreqDropout(1.0, dropout=0.2),
    rfd_5=spec_aug.RandomFreqDropout(1.0, dropout=0.3),
    rfd_6=spec_aug.RandomFreqDropout(1.0, dropout=0.4),
    rfd_7=spec_aug.RandomFreqDropout(1.0, dropout=0.5),
    rfd_8=spec_aug.RandomFreqDropout(1.0, dropout=0.6),
    rfd_9=spec_aug.RandomFreqDropout(1.0, dropout=0.7),
    rfd_10=spec_aug.RandomFreqDropout(1.0, dropout=0.8),
    rfd_11=spec_aug.RandomFreqDropout(1.0, dropout=0.9),
    s_ts_1=signal_aug.TimeStretch(1.0, rate=(0.95, 1.05)),
    s_ts_2=signal_aug.TimeStretch(1.0, rate=(0.9, 1.1)),
    s_ts_3=signal_aug.TimeStretch(1.0, rate=(0.85, 1.15)),
    s_ts_4=signal_aug.TimeStretch(1.0, rate=(0.8, 1.2)),
    s_ts_5=signal_aug.TimeStretch(1.0, rate=(0.75, 1.25)),
    s_psr_1=signal_aug.PitchShiftRandom(1.0, steps=(-1, 1)),
    s_psr_2=signal_aug.PitchShiftRandom(1.0, steps=(-2, 2)),
    s_psr_3=signal_aug.PitchShiftRandom(1.0, steps=(-3, 3)),
    s_psr_4=signal_aug.PitchShiftRandom(1.0, steps=(-4, 4)),
    s_psr_5=signal_aug.PitchShiftRandom(1.0, steps=(-5, 5)),
    # signal augmentation
    s_l_1=signal_aug.Level(1.0, rate=(0.95, 1.05)),
    s_l_2=signal_aug.Level(1.0, rate=(0.9, 1.1)),
    s_l_3=signal_aug.Level(1.0, rate=(0.85, 1.15)),
    s_l_4=signal_aug.Level(1.0, rate=(0.8, 1.2)),
    s_l_5=signal_aug.Level(1.0, rate=(0.75, 1.25)),
    s_o_1=signal_aug.Occlusion(1.0, max_size=0.1),
    s_o_2=signal_aug.Occlusion(1.0, max_size=0.2),
    s_o_3=signal_aug.Occlusion(1.0, max_size=0.3),
    s_o_4=signal_aug.Occlusion(1.0, max_size=0.4),
    s_o_5=signal_aug.Occlusion(1.0, max_size=0.5),
    s_o_6=signal_aug.Occlusion(1.0, max_size=0.6),
    s_o_7=signal_aug.Occlusion(1.0, max_size=0.7),
    s_o_8=signal_aug.Occlusion(1.0, max_size=0.8),
    s_o_9=signal_aug.Occlusion(1.0, max_size=0.9),
    s_o_10=signal_aug.Occlusion(1.0, max_size=1.0),
    s_c_1=signal_aug.Clip(1.0, range=(-0.95, 0.95)),
    s_c_2=signal_aug.Clip(1.0, range=(-0.90, 0.9)),
    s_c_3=signal_aug.Clip(1.0, range=(-0.85, 0.85)),
    s_c_4=signal_aug.Clip(1.0, range=(-0.80, 0.80)),
    s_c_5=signal_aug.Clip(1.0, range=(-0.75, 0.75)),
    s_n_5=signal_aug.Noise(1.0, target_snr=5),
    s_n_10=signal_aug.Noise(1.0, target_snr=10),
    s_n_15=signal_aug.Noise(1.0, target_snr=15),
    s_n_20=signal_aug.Noise(1.0, target_snr=20),
    s_n_25=signal_aug.Noise(1.0, target_snr=25),
    s_n_30=signal_aug.Noise(1.0, target_snr=30),
    s_n_35=signal_aug.Noise(1.0, target_snr=35),
    s_n_40=signal_aug.Noise(1.0, target_snr=40),
)
