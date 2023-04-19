class Chemical:
    """Chemical is a class describing chemicals and/or chemical complexes.
    Chemical.amount: the amount of the Chemical in the cell.
    Chemical.reactions: a list of reactions whose rates are changed if
    Chemical.amount is altered."""

    def __init__(self, amount):
        # Chemical.amount represents the amount of this Chemical in the cell
        self.amount = amount
        # Chemical.reactions is a list of reactions whose rates are changed if
        # Chemical.amount is altered.
        self.reactions = []


class DegradationReaction:
    """DegradationReaction describes the removal of one molecule of
    the specified substrate, with specified rate_constant.
    Overall rate for this reaction to occur is
    substrate.amount * rate_constant."""

    def __init__(self, substrate, rate_constant):
        self.stoichiometry = {substrate: -1}
        self.substrate = substrate
        self.rate_constant = rate_constant
        substrate.reactions.append(self)

    def GetRate(self):
        return self.substrate.amount * self.rate_constant


class CatalyzedSynthesisReaction:
    """CatalyzedSynthesisReaction describes the synthesis of product in'
    the presence of a catalyst:  catalyst -> catalyst + product,
    with specified rate_constant.
    Overall rate for this reaction to occur is
    catalyst.amount * rate_constant."""

    def __init__(self, catalyst, product, rate_constant):
        self.stoichiometry = {product: 1}
        self.catalyst = catalyst
        self.rate_constant = rate_constant
        product.reactions.append(self)

    def GetRate(self):
        return self.catalyst.amount * self.rate_constant


class HeterodimerBindingReaction:
    """HeterodimerBindingReaction describes the binding of two distinct
    types of chemicals, A and B, to form a product dimer: A + B -> dimer,
    with specified rate_constant.
    Overall rate for this reaction to occur is
    A.amount * B.amount * rate_constant."""

    def __init__(self, A, B, dimer, rate_constant):
        self.stoichiometry = {A: -1, B: -1, dimer: 1}
        self.A = A
        self.B = B
        self.rate_constant = rate_constant
        A.reactions.append(self)
        B.reactions.append(self)
        dimer.reactions.append(self)

    def GetRate(self):
        return self.A.amount * self.B.amount * self.rate_constant


class HeterodimerUnbindingReaction:
    """HeterodimerBindingReaction describes the unbinding of a
    heterodimer into two distinct types of chemicals, A and B:
    dimer -> A + B, with specified rate_constant.
    Overall rate for this reaction to occur is
    dimer.amount * rate_constant."""

    def __init__(self, dimer, A, B, rate_constant):
        self.stoichiometry = {A: 1, B: 1, dimer: -1}
        self.dimer = dimer
        self.rate_constant = rate_constant
        A.reactions.append(self)
        B.reactions.append(self)
        dimer.reactions.append(self)

    def GetRate(self):
        return self.dimer.amount * self.rate_constant


class TrimerBindingReaction:
    """
    TrimericBindingReaction describes the binding of three distinct
    types of chemicals, A, B, and C, to form a product trimer: A + B + C -> dimer,
    with specified rate_constant.
    Overall rate for this reaction to occur is
    A.amount * B.amount * rate_constant."""

    def __init__(self, A, B, C, dimer, rate_constant):
        self.stoichiometry = {A: -1, B: -1, C: -1, dimer: 1}
        self.A = A
        self.B = B
        self.C = C
        self.rate_constant = rate_constant
        A.reactions.append(self)
        B.reactions.append(self)
        C.reactions.append(self)
        dimer.reactions.append(self)

    def GetRate(self):
        return self.A.amount * self.B.amount * self.C.amount * self.rate_constant


class HeterodimerDoubleBindingReaction:
    """HeterodimerBindingReaction describes the binding of two distinct
    types of chemicals, A and B, to form a product dimer: A + B -> dimer,
    with specified rate_constant.
    Overall rate for this reaction to occur is
    A.amount * B.amount * rate_constant."""

    def __init__(self, A, B, complex, rate_constant):
        self.stoichiometry = {A: -1, B: -2, complex: 1}
        self.A = A
        self.B = B
        self.rate_constant = rate_constant
        A.reactions.append(self)
        B.reactions.append(self)
        complex.reactions.append(self)

    def GetRate(self):
        return self.A.amount * self.B.amount * (self.B.amount - 1) * self.rate_constant

class HeterodimerDoubleUnbindingReaction:
    """HeterodimerBindingReaction describes the binding of two distinct
    types of chemicals, A and B, to form a product dimer: A + B -> dimer,
    with specified rate_constant.
    Overall rate for this reaction to occur is
    A.amount * B.amount * rate_constant."""

    def __init__(self, A, B, complex, rate_constant):
        self.stoichiometry = {A: 1, B: 2, complex: -1}
        self.complex = complex
        self.rate_constant = rate_constant
        A.reactions.append(self)
        B.reactions.append(self)
        complex.reactions.append(self)

    def GetRate(self):
        return self.complex.amount * self.rate_constant

class TrimerUnbindingReaction:
    """HeterodimerBindingReaction describes the unbinding of a
    trimer into three distinct types of chemicals, A, B and C:
    dimer -> A + B + C, with specified rate_constant.
    Overall rate for this reaction to occur is
    dimer.amount * rate_constant."""

    def __init__(self, dimer, A, B, C, rate_constant):
        self.stoichiometry = {A: 1, B: 1, C: 1, dimer: -1}
        self.dimer = dimer
        self.rate_constant = rate_constant
        A.reactions.append(self)
        B.reactions.append(self)
        C.reactions.append(self)
        dimer.reactions.append(self)

    def GetRate(self):
        return self.dimer.amount * self.rate_constant


class RecombinaseExcisionReaction:
    """
    TrimericBindingReaction describes the binding of three distinct
    types of chemicals, A, B, and C, to form a product trimer: A + B + C -> dimer,
    with specified rate_constant.
    Overall rate for this reaction to occur is
    A.amount * B.amount * rate_constant."""

    def __init__(self, A, B, C, dimer, rate_constant):
        self.stoichiometry = {A: -1, B: -1, C: 0, dimer: 1}
        self.A = A
        self.B = B
        self.C = C
        self.rate_constant = rate_constant
        A.reactions.append(self)
        B.reactions.append(self)
        C.reactions.append(self)
        dimer.reactions.append(self)

    def GetRate(self):
        return self.A.amount * self.B.amount * (1 - self.C.amount) * self.rate_constant
