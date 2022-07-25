import os

import numpy as np

def load_input_file(filename):

    dictionary = { }

    with open(filename, 'r') as f:

        lines = f.readlines()

        for line in lines:
            line = line.lstrip(' ')
            line = line.rstrip(' ')
            if line[0] != '#' and line.count('=') > 0:
                if line.count('#') > 0:
                    line = line.split('#')[0]
                line = line.replace(' ', '').replace('\n', '')
                var, val = line.split('=')

                var = var.lower()
                if val.isnumeric():
                   val = int(val)
                elif val.lower().replace('.','',1).replace('e+','',1).replace('e-','',1).isdigit():
                   val = float(val)
                else:
                    if val.lower() == 'none':
                        val = None
                    elif val.lower() == 'false':
                        val = False
                    elif val.lower() == 'true':
                        val = True
                    elif val.count(';') > 0:
                        if val.count(',') > 0 and val.count('-') == 0:
                            val = [[int(x) for x in v.split(',')] for v in val.split(';')]
                        elif val.count(',') > 0 and val.count('-') > 0:
                            val = [[np.arange(*[int(y)+i for i, y in enumerate(x.split('-'))]).tolist() if x.count('-') > 0 and x[0] != '-' else \
                                   int(x) for x in v.split(',')] if not v.isdigit() else
                                   int(v) for v in val.split(';')]
                        else:
                            val = [int(v) for v in val.split(';')]
                    elif val.count(',') > 0 and val.count('/') == 0:
                        val = [np.arange(*[int(x)+i for i, x in enumerate(v.split('-'))]).tolist() if v.count('-') > 0 and v[0] != '-' else \
                               int(v) if v.isdigit() else \
                               float(v) if v.replace('-','',1).replace('.','',1).isdigit() else \
                               v for v in val.split(',')]
                    elif val.count('-') > 0 and val.count('/') == 0:
                        val = [int(v)+i if v.isdigit() else \
                               #v+'1' if v.isalpha() else \
                               v for i, v in enumerate(val.split('-'))]
                        if type(val[0]) is int:
                            val = np.arange(*val).tolist()
                        elif type(val[0]) is str:
                            val = '-'.join(val)

                dictionary[var] = val

        return dictionary

def output_input_file(filename, directory, outname):

    output_input = os.path.join(directory, outname+'.inp')

    with open(filename, 'r') as f:

        lines = f.readlines()

    with open(output_input, 'w') as f:

        for line in lines:
            if line[0] != '#' and line.count('=') > 0:
                if line.count('#') > 0:
                    line = line.split('#')[0]+'\n'
            f.write(line)

def set_instrument(instrument):

    tof_instruments = ['CORELLI', 'MANDI', 'TOPAZ', 'SNAP']

    instrument = instrument.upper()

    if instrument == 'BL9':
        instrument = 'CORELLI'
    if instrument == 'BL11B':
        instrument = 'MANDI'
    if instrument == 'BL12':
        instrument = 'TOPAZ'
    if instrument == 'BL3':
        instrument = 'SNAP'

    if instrument == 'DEMAND':
        instrument = 'HB3A'
    if instrument == 'WAND2':
        instrument = 'HB2C'

    facility = 'SNS' if instrument in tof_instruments else 'HFIR'

    return facility, instrument

class Experiment:

    def __init__(self, instrument, ipts, run_numbers):

        tof_instruments = ['CORELLI', 'MANDI', 'TOPAZ', 'SNAP']

        instrument = instrument.upper()

        if instrument == 'BL9':
            instrument = 'CORELLI'
        if instrument == 'BL11B':
            instrument = 'MANDI'
        if instrument == 'BL12':
            instrument = 'TOPAZ'
        if instrument == 'BL3':
            instrument = 'SNAP'

        if instrument == 'DEMAND':
            instrument = 'HB3A'
        if instrument == 'WAND2':
            instrument = 'HB2C'

        facility = 'SNS' if instrument in tof_instruments else 'HFIR'

        self.facility, self.instrument = facility, instrument

        self.ipts = ipts

    def get_nexus_file(self, run, exp=None):

        if self.instrument == 'HB3A':
            filepath = '/{}/{}/IPTS-{}/shared/autoreduce/'
            filename = '{}_exp{:04}_scan{:04}.nxs'
        else:
           filepath = '/{}/{}/IPTS-{}/nexus/'
           filename = '{}_{}.nxs.h5'

        filepath = filepath.format(self.facility,self.instrument,self.ipts)
        filename = filename.format(self.instrument,run,exp)

        return os.path.join(filepath,filename)

    def get_output_workspace(self, run, app=None):

        if self.instrument == 'HB2C':
            ows = '{}_{}_{}'.format(self.instrument,run,app)
        if self.instrument == 'HB3A':
            ows = '{}_{}_{}'.format(self.instrument,app,run)
        else:
            ows = '{}_{}'.format(self.instrument,run)

        return ows

    def get_event_workspace(self, run, app=None):

        return self.get_output_workspace(run, app)+'_md'

    def get_peaks_workspace(self, run, app=None):

        return self.get_output_workspace(run, app)+'_pk'