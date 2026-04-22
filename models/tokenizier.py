import re
from models.SPACE_GROUPS import SPACE_GROUPS

ATOMS = [
    "Si", "C", "Pb", "I", "Br", "Cl", "Eu", "O", "Fe", "Sb", "In", "S", "N", "U", "Mn", "Lu", "Se", "Tl", "Hf",
    "Ir", "Ca", "Ta", "Cr", "K", "Pm", "Mg", "Zn", "Cu", "Sn", "Ti", "B", "W", "P", "H", "Pd", "As", "Co", "Np",
    "Tc", "Hg", "Pu", "Al", "Tm", "Tb", "Ho", "Nb", "Ge", "Zr", "Cd", "V", "Sr", "Ni", "Rh", "Th", "Na", "Ru",
    "La", "Re", "Y", "Er", "Ce", "Pt", "Ga", "Li", "Cs", "F", "Ba", "Te", "Mo", "Gd", "Pr", "Bi", "Sc", "Ag", "Rb",
    "Dy", "Yb", "Nd", "Au", "Os", "Pa", "Sm", "Be", "Ac", "Xe", "Kr", "He", "Ne", "Ar",
    # FIX: removed duplicate "Mo"
]

KEYWORDS = [
    "_cell_length_b",
    "_atom_site_occupancy",
    "_atom_site_attached_hydrogens",
    "_cell_length_a",
    "_cell_angle_beta",
    "_symmetry_equiv_pos_as_xyz",
    "_cell_angle_gamma",
    "_atom_site_fract_x",
    "_symmetry_space_group_name_H-M",
    "_symmetry_Int_Tables_number",
    "_chemical_formula_structural",
    "_chemical_name_systematic",
    "_atom_site_fract_y",
    "_atom_site_symmetry_multiplicity",
    "_chemical_formula_sum",
    "_atom_site_label",
    "_atom_site_type_symbol",
    "_cell_length_c",
    "_atom_site_B_iso_or_equiv",
    "_symmetry_equiv_pos_site_id",
    "_cell_volume",
    "_atom_site_fract_z",
    "_cell_angle_alpha",
    "_cell_formula_units_Z",
    "loop_",
    "data_",
]

EXTENDED_KEYWORDS = [
    "_atom_type_symbol",
    "_atom_type_electronegativity",
    "_atom_type_radius",
    "_atom_type_ionic_radius",
    "_atom_type_oxidation_number",
    "_prop_is_metal",
    "_prop_fermi_energy",
    "_prop_fermi_energy_units",
    "_prop_material_id",
    "_prop_formula",
    "_prop_band_gap",
    "_prop_band_gap_units",
    "_prop_band_gap_method",
    "_prop_band_gap_direct",
    "_prop_is_magnetic",
    "_prop_total_magnetization",
    "_prop_total_magnetization_units",
    "_prop_energy_above_hull",
    "_prop_energy_above_hull_units",
    "_prop_formation_energy_per_atom",
    "_prop_formation_energy_units",
    "_prop_is_stable",
    "_prop_density",
    "_prop_density_units",
    "_prop_volume",
    "_prop_volume_units",
    "_prop_nsites",
    "_prop_crystal_system",
    "_prop_space_group",
    "_prop_space_group_number",
]

DIGITS = [str(i) for i in range(10)]

# Special tokens for model training
UNK_TOKEN = "<unk>"
PAD_TOKEN = "<pad>"
BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"
SPECIAL_TOKENS = [PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, UNK_TOKEN]


class CIFTokenizer:
    def __init__(self, add_special_tokens=True):
        """
        Args:
            add_special_tokens: If True, adds <pad>, <bos>, <eos>, <unk> tokens.
                                 Useful for model training. Set False for plain tokenization.
        """
        self._add_special_tokens = add_special_tokens

        self._tokens = []

        # Special tokens go first so <pad>=0 is conventional
        if add_special_tokens:
            self._tokens.extend(SPECIAL_TOKENS)

        self._tokens.extend(self.atoms())
        self._tokens.extend(self.digits())
        self._tokens.extend(self.keywords())
        self._tokens.extend(self.symbols())

        # Disambiguated space groups (e.g. 'Pm' -> 'Pm_sg')
        space_groups = list(self.space_groups())
        space_groups_sg = [sg + "_sg" for sg in space_groups]
        self._tokens.extend(space_groups_sg)

        # Build vocab mappings
        self._token_to_id = {tok: i for i, tok in enumerate(self._tokens)}
        self._id_to_token = {i: tok for i, tok in enumerate(self._tokens)}

        # Decode space group tokens back to their real symbol (without _sg)
        for sg in space_groups_sg:
            self._id_to_token[self._token_to_id[sg]] = sg.replace("_sg", "")

        # Regex: sorted longest-first to avoid partial matches
        core_tokens = [t for t in self._tokens if not t.startswith("<")]
        escaped = sorted([re.escape(t) for t in core_tokens], key=len, reverse=True)
        token_pattern = "|".join(escaped)
        self._full_pattern = f"({token_pattern}|\\w+|[\\.,;!?])"

    # ------------------------------------------------------------------ #
    #  Vocabulary sources                                                  #
    # ------------------------------------------------------------------ #

    @staticmethod
    def atoms():
        return list(ATOMS)

    @staticmethod
    def digits():
        return list(DIGITS)

    @staticmethod
    def keywords():
        kws = list(KEYWORDS)
        kws.extend(EXTENDED_KEYWORDS)
        return kws

    @staticmethod
    def symbols():
        return ["x", "y", "z", ".", "(", ")", "+", "-", "/", "'", ",", " ", "\n"]

    @staticmethod
    def space_groups():
        return SPACE_GROUPS

    # ------------------------------------------------------------------ #
    #  Public properties                                                   #
    # ------------------------------------------------------------------ #

    @property
    def vocab_size(self):
        return len(self._tokens)

    @property
    def token_to_id(self):
        return dict(self._token_to_id)

    @property
    def id_to_token(self):
        return dict(self._id_to_token)

    @property
    def pad_token_id(self):
        return self._token_to_id.get(PAD_TOKEN)

    @property
    def bos_token_id(self):
        return self._token_to_id.get(BOS_TOKEN)

    @property
    def eos_token_id(self):
        return self._token_to_id.get(EOS_TOKEN)

    @property
    def unk_token_id(self):
        return self._token_to_id[UNK_TOKEN]

    # ------------------------------------------------------------------ #
    #  Core methods                                                        #
    # ------------------------------------------------------------------ #

    def tokenize_cif(self, cif_string, single_spaces=True):
        """Convert a raw CIF string into a list of string tokens."""
        # Disambiguate space group symbols from atom names
        spacegroups = "|".join(SPACE_GROUPS)
        cif_string = re.sub(
            fr'(_symmetry_space_group_name_H-M *\b({spacegroups}))\n',
            r'\1_sg\n',
            cif_string,
        )

        if single_spaces:
            cif_string = re.sub(r'[ \t]+', ' ', cif_string)

        tokens = re.findall(self._full_pattern, cif_string)

        # Replace unrecognized tokens with <unk>
        valid = set(self._tokens)
        tokens = [t if t in valid else UNK_TOKEN for t in tokens]

        return tokens

    def encode(self, tokens, add_special_tokens=False):
        """
        Convert a list of string tokens to a list of integer IDs.

        Args:
            tokens: list of string tokens (from tokenize_cif)
            add_special_tokens: if True, wraps with <bos> and <eos> IDs
        """
        ids = [self._token_to_id.get(t, self._token_to_id[UNK_TOKEN]) for t in tokens]
        if add_special_tokens and self._add_special_tokens:
            ids = [self.bos_token_id] + ids + [self.eos_token_id]
        return ids

    def decode(self, ids, skip_special_tokens=True):
        """
        Convert a list of integer IDs back to a CIF string.

        Args:
            ids: list of integer token IDs
            skip_special_tokens: if True, strips <pad>/<bos>/<eos>/<unk>
        """
        special = {PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, UNK_TOKEN}
        tokens = []
        for i in ids:
            tok = self._id_to_token.get(i, UNK_TOKEN)
            if skip_special_tokens and tok in special:
                continue
            tokens.append(tok)
        return "".join(tokens)

    def tokenize_and_encode(self, cif_string, single_spaces=True, add_special_tokens=False):
        """Convenience method: tokenize then encode in one call."""
        tokens = self.tokenize_cif(cif_string, single_spaces=single_spaces)
        return self.encode(tokens, add_special_tokens=add_special_tokens)