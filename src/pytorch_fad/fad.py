from .fad_score import compute_statistics_of_path, calculate_frechet_distance
from .vggish import get_vggish_model
import os

#batch size and num workers currently unused
class FADMetric:
    def __init__(self, device='cuda', base_path=None, recursive=True, batch_size=1, num_workers=1,):
        self.model = get_vggish_model(device)
        self.batch_size = 1 #batch_size
        self.num_workers = 1 #num_workers
        self.base_path = base_path
        self.recursive = recursive

        if os.path.exists(self.base_path):
            self.base_m, self.base_s = compute_statistics_of_path(self.base_path, self.model,
                                                                  self.batch_size, self.num_workers,
                                                                  self.recursive)
        else:
            raise RuntimeError('Invalid path: %s' % self.base_path)

    def compare_base_to_path(self, path):
        m, s = compute_statistics_of_path(path, self.model,
                                          self.batch_size, self.num_workers,
                                          self.recursive)
        self.fad_value = calculate_frechet_distance(self.base_m, self.base_s, m, s)

        return self.fad_value

    def compare_paths(self, path1, path2):
        m1, s1 = compute_statistics_of_path(path1, self.model,
                                            self.batch_size, self.num_workers,
                                            self.recursive)
        m2, s2 = compute_statistics_of_path(path2, self.model,
                                            self.batch_size, self.num_workers,
                                            self.recursive)
        self.fad_value = calculate_frechet_distance(m1, s1, m2, s2)

        return self.fad_value
