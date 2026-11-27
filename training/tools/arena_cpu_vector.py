"""Batch uniform Gymnasium info columns for CPU training rollouts."""
import numpy as np
from gymnasium.vector import AsyncVectorEnv


class BatchedInfoVectorEnv(AsyncVectorEnv):
    """Keep Gymnasium's full info layout, including per-key presence masks."""

    def _add_info(self, vector_infos, env_info, env_num):
        if env_num == 0:
            self._step_infos = []
        self._step_infos.append(env_info)
        if env_num + 1 < self.num_envs:
            return vector_infos
        rows = self._step_infos
        self._step_infos = []
        first = rows[0]
        if ("final_info" in first or "final_obs" in first
                or any(row.keys() != first.keys() for row in rows)
                or any(isinstance(value, dict) for value in first.values())):
            # Gymnasium recurses through self._add_info for nested dictionaries.
            return self._standard_info(rows)
        output = {}
        for key, first_value in first.items():
            if isinstance(first_value, np.ndarray) or isinstance(first_value, (bool, int, float, np.number)):
                output[key] = np.asarray([row[key] for row in rows])
            else:
                return self._standard_info(rows)
            output["_" + key] = np.ones(len(rows), dtype=np.bool_)
        return output

    def _standard_info(self, rows):
        # Bind the reference implementation to a plain aggregator for recursive calls.
        from gymnasium.vector import VectorEnv
        owner = VectorEnv()
        owner.num_envs = self.num_envs
        out = {}
        for index, row in enumerate(rows):
            out = owner._add_info(out, row, index)
        return out
