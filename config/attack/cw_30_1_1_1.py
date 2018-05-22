from src.attack.cw import CW

attacker = CW
attacker_args = tuple()
attacker_kwargs = {
    'max_iter': 30,
    'n_restart': 1,
    'max_binary_step': 1,
    'initial_c': 1e0,
    'noprint': True,
}
