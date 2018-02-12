from src.attack.deep_fool import DeepFool

attacker = DeepFool
attacker_args = tuple()
attacker_kwargs = {'overshoot': .02, 'max_iter': 50}
