class MeltDataFrame:
    def __init__(self, *, id_vars, value_vars, var_name, value_name):
        self.id_vars = id_vars
        self.value_vars = value_vars
        self.var_name = var_name
        self.value_name = value_name

    def transform(self, df):
        expr = ", ".join([f"'{col}', {col}" for col in self.value_vars])
        return df.selectExpr(
            *self.id_vars,
            f"stack({len(self.value_vars)}, {expr}) as ({self.var_name}, {self.value_name})",
        )
