from __future__ import print_function

def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std

def adaptive_channel(student_feat, teacher_feat):
    # assert (student_feat.size()[:2] == teacher_feat.size()[:2])
    size = student_feat.size()
    t_mean, t_std = calc_mean_std(teacher_feat)
    s_mean, s_std = calc_mean_std(student_feat)
    normalized_feat = (student_feat - s_mean.expand(size)) / s_std.expand(size)
    return normalized_feat * t_std.expand(size) + t_mean.expand(size)


