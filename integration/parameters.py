import numpy as np

def load_input_file(filename):

    dictionary = { }

    with open(filename, 'r') as f:
        
        lines = f.readlines()
        
        for line in lines:
            if line[0] != '#' and line.count('=') > 0:
                if line.count('#') > 0:
                    line, _ = line.split('#')
                line = line.replace(' ', '').replace('\n', '')
                var, val = line.split('=')
                
                var = var.lower()
                if val.isnumeric():
                   val = int(val) if val.isdigit() else float(val)
                else:
                    if val.lower() == 'none':
                        val = None
                    elif val.lower() == 'false':
                        val = False
                    elif val.lower() == 'true':
                        val = True
                    elif val.count(',') > 0:
                        val = [int(v) if v.isdigit() else \
                               float(v) if v.isdecimal() else \
                               np.arange(*[int(x)+i for i, x in enumerate(v.split('-'))]).tolist() if v.count('-') > 0 else \
                               v for v in val.split(',')]
                    elif val.count('-') > 0:
                        val = [int(v)+i if v.isdigit() else \
                               v+'1' if v.isalpha() else \
                               v for i, v in enumerate(val.split('-'))]
                        if type(val[0]) is int:
                            val = np.arange(*val).tolist()
                            
                dictionary[var] = val

        return dictionary