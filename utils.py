import torch
import numpy as np
import torch.nn as nn
import gym
import os
from collections import deque
import random
# BEGIN
# dependencies for AE priority sampling and replay snapshots
import re
from replay_priority import SumTree, index_array, positive_integer
# END


class eval_mode(object):
    def __init__(self, *models):
        self.models = models

    def __enter__(self):
        self.prev_states = []
        for model in self.models:
            self.prev_states.append(model.training)
            model.train(False)

    def __exit__(self, *args):
        for model, state in zip(self.models, self.prev_states):
            model.train(state)
        return False


def soft_update_params(net, target_net, tau):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(
            tau * param.data + (1 - tau) * target_param.data
        )


def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def module_hash(module):
    result = 0
    for tensor in module.state_dict().values():
        result += tensor.sum().item()
    return result


def make_dir(dir_path):
    try:
        os.mkdir(dir_path)
    except OSError:
        pass
    return dir_path


def preprocess_obs(obs, bits=5):
    """Preprocessing image, see https://arxiv.org/abs/1807.03039."""
    bins = 2**bits
    assert obs.dtype == torch.float32
    if bits < 8:
        obs = torch.floor(obs / 2**(8 - bits))
    obs = obs / bins
    obs = obs + torch.rand_like(obs) / bins
    obs = obs - 0.5
    return obs


class ReplayBuffer(object):
    """Buffer to store environment transitions."""
    # BEGIN
    # backward-compatible AE priority configuration and metadata
    def __init__(self, obs_shape, action_shape, capacity, batch_size, device,
                 ae_priority_alpha=0.7, ae_priority_eps=1e-2, ae_seed=0,
                 ae_batch_size=None):
        capacity = positive_integer(capacity, 'capacity')
        self.capacity = capacity
        self.batch_size = positive_integer(batch_size, 'batch_size')
        # BEGIN
        # retain an independent AE default, inheriting RL size if omitted
        self.ae_batch_size = positive_integer(
            self.batch_size if ae_batch_size is None else ae_batch_size, 'ae_batch_size'
        )
        # END
        self.device = device
        self.ae_priority_alpha, self.ae_priority_eps = self._priority_parameters(
            ae_priority_alpha, ae_priority_eps
        )
        seed = index_array([ae_seed], 2**32)[0]
        self._ae_rng = np.random.RandomState(seed)
        # BEGIN
        # match the reference epsilon-based initial sampling mass
        # Insert eps directly, as in the ZIP; only uncertainty updates use powers.
        # With alpha=0, all initial and updated masses must instead remain one.
        self._ae_max_mass = self._initial_ae_mass()
        # END
        self._ae_tree = SumTree(capacity)

        # the proprioceptive obs is stored as float32, pixels obs as uint8
        obs_dtype = np.float32 if len(obs_shape) == 1 else np.uint8

        self.obses = np.empty((capacity, *obs_shape), dtype=obs_dtype)
        self.next_obses = np.empty((capacity, *obs_shape), dtype=obs_dtype)
        self.actions = np.empty((capacity, *action_shape), dtype=np.float32)
        self.rewards = np.empty((capacity, 1), dtype=np.float32)
        self.not_dones = np.empty((capacity, 1), dtype=np.float32)

        self.idx = 0
        self.last_save = 0
        self.full = False
    # END

    # BEGIN
    # populated size and validation of AE priority parameters
    def __len__(self):
        return self.capacity if self.full else self.idx

    @staticmethod
    def _priority_parameters(alpha, eps):
        result = []
        for name, value in [('ae_priority_alpha', alpha), ('ae_priority_eps', eps)]:
            value = np.asarray(value)
            if value.ndim != 0 or value.dtype.kind not in 'iuf':
                raise ValueError('%s must be a finite number' % name)
            number = float(value)
            if not np.isfinite(number) or number < 0 or (name.endswith('eps') and number == 0):
                raise ValueError('%s is outside its valid range' % name)
            result.append(number)
        return tuple(result)
    # END

    # BEGIN
    # shared initialization rule for new and legacy replay data
    def _initial_ae_mass(self):
        mass = 1.0 if self.ae_priority_alpha == 0 else self.ae_priority_eps
        if mass > np.finfo(np.float64).max / (2 * self.capacity):
            raise ValueError('initial AE priority is too large for buffer capacity')
        return mass
    # END

    def add(self, obs, action, reward, next_obs, done):
        np.copyto(self.obses[self.idx], obs)
        np.copyto(self.actions[self.idx], action)
        np.copyto(self.rewards[self.idx], reward)
        np.copyto(self.next_obses[self.idx], next_obs)
        np.copyto(self.not_dones[self.idx], not done)

        # BEGIN
        # initialize or replace the priority at this exact replay slot
        self._ae_tree.update_batch([self.idx], [self._ae_max_mass])
        # END
        self.idx = (self.idx + 1) % self.capacity
        self.full = self.full or self.idx == 0

    def sample(self):
        idxs = np.random.randint(
            0, self.capacity if self.full else self.idx, size=self.batch_size
        )

        obses = torch.as_tensor(self.obses[idxs], device=self.device).float()
        actions = torch.as_tensor(self.actions[idxs], device=self.device)
        rewards = torch.as_tensor(self.rewards[idxs], device=self.device)
        next_obses = torch.as_tensor(
            self.next_obses[idxs], device=self.device
        ).float()
        not_dones = torch.as_tensor(self.not_dones[idxs], device=self.device)

        return obses, actions, rewards, next_obses, not_dones

    # BEGIN
    # independent AE sampling and uncertainty-to-mass updates
    # BEGIN
    # use UCRL-C stratified sampling with possible repeated AE transitions
    # BEGIN
    # Optionally return the additional fields needed for ensemble TD scoring.
    def sample_ae(self, batch_size=None, *, include_transition=False):
        """Draw one transition per equal-mass segment; indices stay on CPU.

        Repeated transitions are kept and priority masses do not change during
        sampling. Match UCRL-C's population guard even though repeats are allowed:
        batch size must not exceed the populated size. No crops, normalization,
        or importance weights are applied. Sampling preserves priority state.
        The stored actions are returned for critic-disagreement scoring.
        With include_transition, return (obs, action, reward, next_obs,
        not_done, indices) for the same draw.
        Omitted sizes use the configured AE default; explicit sizes override
        only this call and do not change either stored batch size.
        """
        # END
        # BEGIN
        # use the AE batch default without changing RL sampling
        batch_size = self.ae_batch_size if batch_size is None else positive_integer(
            batch_size, 'ae_batch_size'
        )
        # END
        if len(self) == 0:
            raise ValueError('cannot sample AE transitions from an empty buffer')
        if batch_size > len(self):
            raise ValueError('ae_batch_size exceeds the number of stored transitions')
        rng_state = self._ae_rng.get_state()
        try:
            indices = self._ae_tree.sample_stratified(batch_size, self._ae_rng)
            obs = torch.as_tensor(self.obses[indices], device=self.device).float()
            actions = torch.as_tensor(self.actions[indices], device=self.device)
            # BEGIN
            # Keep these tensors aligned with the AE indices, including repeats.
            if include_transition:
                rewards = torch.as_tensor(self.rewards[indices], device=self.device)
                next_obs = torch.as_tensor(self.next_obses[indices], device=self.device).float()
                not_dones = torch.as_tensor(self.not_dones[indices], device=self.device)
                return obs, actions, rewards, next_obs, not_dones, indices
            # END
        except BaseException:
            # Tensor construction can fail after sampling has consumed private RNG.
            self._ae_rng.set_state(rng_state)
            raise
        return obs, actions, indices
    # END

    def update_ae_priorities(self, indices, uncertainty):
        # BEGIN
        # Both estimators return raw nonnegative scores.
        """Accept raw AE priority scores; add epsilon and exponentiate once.

        Invalid batches leave all priority state unchanged. Duplicate indices
        use their last supplied uncertainty. As in UCRL-C, the historical
        maximum considers all transformed scores, including earlier duplicates.
        """
        # END
        if torch.is_tensor(indices):
            indices = indices.detach().cpu().numpy()
        if torch.is_tensor(uncertainty):
            uncertainty = uncertainty.detach().cpu().numpy()
        indices = index_array(indices, len(self))
        uncertainty = np.asarray(uncertainty)
        if uncertainty.ndim == 2 and uncertainty.shape[1] == 1:
            uncertainty = uncertainty[:, 0]
        if uncertainty.ndim != 1 or uncertainty.dtype.kind not in 'iuf':
            raise ValueError('uncertainty must be a numeric vector or column')
        if len(indices) != len(uncertainty):
            raise ValueError('indices and uncertainty must have the same length')
        uncertainty = uncertainty.astype(np.float64)
        if not np.all(np.isfinite(uncertainty)) or np.any(uncertainty < 0):
            raise ValueError('uncertainty must be finite and nonnegative')
        if len(indices) == 0:
            return
        with np.errstate(over='ignore', under='ignore', invalid='ignore'):
            masses = (uncertainty + self.ae_priority_eps) ** self.ae_priority_alpha
        if not np.all(np.isfinite(masses)) or np.any(masses <= 0):
            raise ValueError('transformed priorities must be finite and positive')
        # Reserve headroom so even filling every slot at the maximum stays finite.
        if np.any(masses > np.finfo(np.float64).max / (2 * self.capacity)):
            raise ValueError('transformed priorities are too large for buffer capacity')
        self._ae_tree.update_batch(indices, masses)
        # BEGIN
        # preserve UCRL-C's historical maximum even when a duplicate is overwritten
        self._ae_max_mass = max(self._ae_max_mass, float(masses.max()))
        # END
    # END

    # BEGIN
    # complete replay snapshots with legacy chunk compatibility
    @staticmethod
    def _read_replay(path, legacy=False):
        # Legacy chunks contain NumPy pickles: load only trusted local replay files.
        try:
            return torch.load(path, map_location='cpu', weights_only=not legacy)
        except TypeError as error:
            # Keep compatibility with the older PyTorch used by this repository.
            if 'weights_only' not in str(error):
                raise
            return torch.load(path, map_location='cpu')

    def _transition_arrays(self):
        return (self.obses, self.next_obses, self.actions, self.rewards, self.not_dones)

    def save(self, save_dir):
        """Atomically replace a complete snapshot, including AE RNG and masses.

        Save even when idx is unchanged: an entire ring may have been replaced,
        or priorities/RNG may have changed without any environment interaction.
        """
        rng = self._ae_rng.get_state()
        payload = dict(
            version=1, capacity=self.capacity, idx=self.idx, full=self.full,
            transitions=[torch.from_numpy(a[:len(self)]) for a in self._transition_arrays()],
            ae_priority_alpha=self.ae_priority_alpha,
            ae_priority_eps=self.ae_priority_eps,
            ae_masses=torch.from_numpy(self._ae_tree.masses[:len(self)]),
            ae_max_mass=self._ae_max_mass,
            ae_rng=dict(algorithm=rng[0], keys=torch.from_numpy(rng[1].astype(np.int64)),
                        position=rng[2], has_gauss=rng[3], cached_gaussian=rng[4])
        )
        path = os.path.join(save_dir, 'replay_buffer.pt')
        temporary = path + '.tmp'
        try:
            with open(temporary, 'wb') as handle:
                torch.save(payload, handle)
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.remove(temporary)
        self.last_save = self.idx

    def load(self, save_dir):
        """Restore a snapshot, or initialize priorities for old chunk files.

        Configuration and AE RNG are restored from new snapshots. Legacy files
        use this buffer's configuration and RNG because they contain neither.
        Validation finishes before the live buffer is changed.
        """
        path = os.path.join(save_dir, 'replay_buffer.pt')
        if os.path.isfile(path):
            payload = self._read_replay(path)
            if not isinstance(payload, dict) or payload.get('version') != 1:
                raise ValueError('unsupported replay snapshot version')
            if payload.get('capacity') != self.capacity:
                raise ValueError('snapshot capacity differs from buffer capacity')
            idx, full = payload['idx'], payload['full']
            if isinstance(idx, bool) or not isinstance(idx, int) or not 0 <= idx < self.capacity:
                raise ValueError('invalid snapshot ring position')
            if not isinstance(full, bool):
                raise ValueError('invalid snapshot full flag')
            size = self.capacity if full else idx
            arrays = payload['transitions']
            alpha, eps = self._priority_parameters(
                payload['ae_priority_alpha'], payload['ae_priority_eps']
            )
            masses = payload['ae_masses'].numpy()
            if masses.shape != (size,) or np.any(masses <= 0):
                raise ValueError('invalid snapshot AE masses')
            max_mass = float(payload['ae_max_mass'])
            if (not np.isfinite(max_mass) or max_mass <= 0
                    or max_mass < (float(masses.max()) if size else 0.0)
                    or max_mass > np.finfo(np.float64).max / (2 * self.capacity)):
                raise ValueError('invalid snapshot maximum mass')
            tree = SumTree.from_masses(self.capacity, masses)
            state = payload['ae_rng']
            keys = state['keys'].numpy()
            if (keys.shape != (624,) or keys.dtype.kind not in 'iu'
                    or np.any(keys < 0) or np.any(keys >= 2**32)
                    or state['algorithm'] != 'MT19937'
                    or not isinstance(state['position'], int) or not 0 <= state['position'] <= 624
                    or state['has_gauss'] not in (0, 1)
                    or not np.isfinite(state['cached_gaussian'])):
                raise ValueError('invalid snapshot AE RNG state')
            rng = np.random.RandomState(0)
            rng.set_state((state['algorithm'], keys.astype(np.uint32),
                           state['position'], state['has_gauss'], state['cached_gaussian']))
        else:
            chunks = []
            for name in os.listdir(save_dir):
                match = re.fullmatch(r'(\d+)_(\d+)\.pt', name)
                if match:
                    chunks.append((int(match[1]), int(match[2]), name))
            if not chunks:
                raise ValueError('no replay snapshot or legacy chunks found')
            fields = [[] for _ in self._transition_arrays()]
            size = 0
            for start, end, name in sorted(chunks):
                if start != size or not start < end <= self.capacity:
                    raise ValueError('legacy chunks must be contiguous and fit capacity')
                chunk = self._read_replay(os.path.join(save_dir, name), legacy=True)
                chunk = self._validated_arrays(chunk, end - start)
                for parts, array in zip(fields, chunk):
                    parts.append(array)
                size = end
            arrays = [np.concatenate(parts, axis=0) for parts in fields]
            full, idx = size == self.capacity, size % self.capacity
            alpha, eps = self.ae_priority_alpha, self.ae_priority_eps
            # BEGIN
            # legacy data uses the same initial mass as new insertions
            max_mass = self._initial_ae_mass()
            tree = SumTree.from_masses(self.capacity, np.full(size, max_mass))
            # END
            rng = self._ae_rng

        arrays = self._validated_arrays(arrays, size)
        for destination, source in zip(self._transition_arrays(), arrays):
            np.copyto(destination[:size], source)
        self.idx, self.full, self.last_save = idx, full, idx
        self.ae_priority_alpha, self.ae_priority_eps = alpha, eps
        self._ae_tree, self._ae_max_mass, self._ae_rng = tree, max_mass, rng

    def _validated_arrays(self, arrays, size):
        if not isinstance(arrays, (list, tuple)) or len(arrays) != 5:
            raise ValueError('replay must contain five transition arrays')
        result = []
        for array, destination in zip(arrays, self._transition_arrays()):
            if torch.is_tensor(array):
                array = array.numpy()
            array = np.asarray(array)
            if array.shape != (size, *destination.shape[1:]) or array.dtype != destination.dtype:
                raise ValueError('replay transition shape or dtype mismatch')
            result.append(array)
        return result
    # END


class FrameStack(gym.Wrapper):
    def __init__(self, env, k):
        gym.Wrapper.__init__(self, env)
        self._k = k
        self._frames = deque([], maxlen=k)
        shp = env.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=0,
            high=1,
            shape=((shp[0] * k,) + shp[1:]),
            dtype=env.observation_space.dtype
        )
        self._max_episode_steps = env._max_episode_steps

    def reset(self):
        obs = self.env.reset()
        for _ in range(self._k):
            self._frames.append(obs)
        return self._get_obs()

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self._frames.append(obs)
        return self._get_obs(), reward, done, info

    def _get_obs(self):
        assert len(self._frames) == self._k
        return np.concatenate(list(self._frames), axis=0)
