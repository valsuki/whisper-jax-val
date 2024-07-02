
from whisper_jax import FlaxWhisperForConditionalGeneration, FlaxWhisperPipline
import jax.numpy as jnp

checkpoint_id = "sanchit-gandhi/whisper-small-hi"
# convert PyTorch weights to Flax
model = FlaxWhisperForConditionalGeneration.from_pretrained(checkpoint_id, from_pt=True)
# push converted weights to the Hub
model.push_to_hub(checkpoint_id)

# now we can load the Flax weights directly as required
pipeline = FlaxWhisperPipline(checkpoint_id, dtype=jnp.bfloat16, batch_size=1)

