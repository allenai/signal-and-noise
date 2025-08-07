def _parse_choices(response: str, n_choices: int, sep='-'):
    n_choices = [n_choices] if not isinstance(n_choices, list) else n_choices
    parsed = None
    try:
        parsed = response.split(f'\n{sep} ')
        parsed[0] = f'{sep} '.join(parsed[0].split(f'{sep} ')[1:])

        # If we extracted too many answers, remove empty answers
        if len(parsed) > max(n_choices): 
            parsed = [choice for choice in parsed if choice != '']

        assert len(parsed) in n_choices, f'Response length: {len(parsed)}, requested length: {n_choices}'
    except (IndexError, AttributeError, AssertionError) as e:
        print(f"Error parsing response: {e}\nResponse:\n{response}\nParsed choices: {parsed}")
        parsed = None
    
    # remove trailing spaces from responses
    if parsed is not None:
        parsed = [choice.rstrip() for choice in parsed]
    
    return parsed