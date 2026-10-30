"""Identity processing for the generic RLT policy's explicitly mapped observations."""
from lerobot.processor import PolicyProcessorPipeline, batch_to_transition, transition_to_batch, policy_action_to_transition, transition_to_policy_action

def make_rlt_pre_post_processors(config, dataset_stats=None):
    return (PolicyProcessorPipeline(steps=[], to_transition=batch_to_transition, to_output=transition_to_batch),
            PolicyProcessorPipeline(steps=[], to_transition=policy_action_to_transition, to_output=transition_to_policy_action))
