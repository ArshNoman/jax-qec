def simulate_qec_cycle(logical_state, code, noise_model, decoder, key):
    """
    Simulate a full QEC cycle:
    - Encode logical state
    - Apply noise
    - Measure syndrome
    - Decode
    - Return corrected state

    Parameters:
    - logical_state: jnp.ndarray, shape (2,) for |0⟩ or |1⟩
    - code: instance of RepetitionEncode or similar
    - noise_model: instance of BitFlipNoise, PhaseFlipNoise, etc.
    - decoder: instance of a Decoder subclass
    - key: JAX PRNGKey

    Returns:
    - corrected_state: jnp.ndarray, shape (2^n,)
    """
    encoded = code.encode(logical_state)
    noisy = noise_model.apply(encoded, key)
    collapsed = code.decode_collapsed(noisy)
    syndrome = code.measure_syndrome_collapsed(collapsed)
    corrected = decoder.decode(collapsed, syndrome)
    return corrected
