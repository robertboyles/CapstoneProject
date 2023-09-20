from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback, CallbackList, EvalCallback
import numpy as np
import os

class TensorboardLapTime(BaseCallback):
    def __init__(self, control_freq=10.0, verbose: int = 0):
        super().__init__(verbose)
        self.sf = control_freq
    def _on_rollout_end(self) -> None:

        laptime_if_success = np.array(
            [ep_info["l"] * succ * 1/self.sf for ep_info, succ in zip(self.model.ep_info_buffer, self.model.ep_success_buffer)])
        laptime_if_success = laptime_if_success[laptime_if_success != 0]

        mean_lap_time = 0.0 if laptime_if_success.size == 0 else np.mean(laptime_if_success)
        min_lap_time = 0.0 if laptime_if_success.size == 0 else np.min(laptime_if_success)

        self.logger.record("rollout/lap_time_av", mean_lap_time)
        self.logger.record("rollout/lap_time_min", min_lap_time)

    def _on_step(self) -> bool:
        return True

class MinimumLapTimeSave(BaseCallback):
    def __init__(self, control_freq=10.0, verbose: int = 0):
        super().__init__(verbose)
        self.sf = control_freq
        self.best_lap_time = 1000000.0
    def _on_step(self) -> None:
        env_best = self.model.env.envs[0].minimum_observed_laptime
        if env_best is not None and env_best <= self.best_lap_time \
            and self.parent.best_model_save_path is not None:
            print('Saving best lap time model. Time : %.4f' % env_best)
            self.model.save(os.path.join(self.parent.best_model_save_path, "best_laptime"))
            


def callbackset(save_path, save_name, save_freq, control_freq, eval_path, best_save_path, env, eval_every):
    checkpoint_callback = CheckpointCallback(
    save_freq=save_freq,
    save_path=save_path,
    name_prefix=save_name
    )
    eval_call_back = EvalCallback(env, None, MinimumLapTimeSave(control_freq=control_freq, verbose=0), 1, eval_every, eval_path, best_save_path, True, False, 1)
    callbacks = CallbackList([TensorboardLapTime(control_freq=control_freq), checkpoint_callback, eval_call_back])
    return callbacks