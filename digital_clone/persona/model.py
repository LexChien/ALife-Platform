class PersonaModel:
    def __init__(self, name, tone, principles, goals, facts=None):
        self.name = name
        self.tone = tone
        self.principles = principles
        self.goals = goals
        self.facts = facts or []
