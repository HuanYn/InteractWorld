"""Small authored skill contracts, not learned programs or model successes."""

SKILLS = [
    dict(name='retrieve_key', condition='route B or no route; independent investigation authorized',
         sequence=['meet Anna if needed', 'choose independent investigation if needed', 'search desk', 'pick up red_key'],
         completion='red_key in rule inventory; visual pickup assessed separately',
         failure='incompatible route / key unavailable / visual uncertainty'),
    dict(name='negotiate_password', condition='route A or no route; helping Anna authorized',
         sequence=['meet Anna if needed', 'help Anna once', 'ask password'],
         completion='password_known rule flag; dialogue not inferred from silent frames',
         failure='incompatible route / trust insufficient'),
    dict(name='unlock_terminal', condition='route A and password_known',
         sequence=['use terminal'], completion='terminal unlocked rule state',
         failure='missing password / wrong route / visual uncertainty'),
]


def skill_context(state):
    route = state.story.route
    allowed = {'retrieve_key': route in (None, 'B'),
               'negotiate_password': route in (None, 'A'),
               'unlock_terminal': route == 'A' and bool(state.story.flags.get('password_known'))}
    return [dict(skill, currently_applicable=allowed[skill['name']], source='authored_contract_not_learned')
            for skill in SKILLS]
