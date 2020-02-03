import base64
import cffi
import os
import sys
import time

def find(name, root, *subdirs):
    paths = [os.path.join(root, name)]
    paths += [os.path.join(root, d, name) for d in subdirs]
    for path in paths:
        if os.path.exists(path):
            return path
    raise FileNotFoundError('could not find {} in {}'.format(name, root))

ffi = None
lib = None
def load_w2l(path):
    global ffi, lib
    if lib is None:
        if os.name == 'nt':
            w2l_library = 'libw2l.dll'
        elif os.name == 'posix':
            if sys.platform == 'darwin':
                w2l_library = 'libw2l.dylib'
            else:
                w2l_library = 'libw2l.so'
        else:
            raise RuntimeError('unsupported OS')

        w2l_h = find('w2l.h', path, 'include')
        w2l_lib = find(w2l_library, path, 'lib')

        ffi = cffi.FFI()
        ffi.cdef('void free(void *);')
        lines = []
        ifdefs = 0
        with open(w2l_h, 'r') as f:
            for line in f.read().split('\n'):
                line = line.strip()
                if line.startswith('#ifdef'):
                    ifdefs += 1
                elif line.startswith('#endif'):
                    ifdefs -= 1
                elif ifdefs == 0 and not line.startswith('#'):
                    lines.append(line)
        header = '\n'.join(lines)
        ffi.cdef(header)
        lib = ffi.dlopen(w2l_lib)

def consume_c_text(c_text, sep):
    if not c_text:
        return []
    text = ffi.string(c_text).decode('utf8')
    lib.free(c_text)
    if not text:
        return []
    return text.strip().split(sep)

class W2lEncoder:
    def __init__(self, am, tokens):
        if not os.path.exists(am):
            raise FileNotFoundError(am)
        if not os.path.exists(tokens):
            raise FileNotFoundError(tokens)

        self.am = am
        self.tokens = tokens
        self.handle = None
        self.handle = lib.w2l_engine_new(am.encode('utf8'), tokens.encode('utf8'))

    def emit(self, samples):
        array = ffi.new('float []', list(samples))
        emission = lib.w2l_engine_process(self.encoder.handle, array, len(array))
        try:
            emit_text = lib.w2l_emission_text(emission)
            emit = consume_c_text(emit_text, sep=' ')
            return emit
        finally:
            lib.w2l_emission_free(emission)

    def __del__(self):
        if self.handle:
            lib.w2l_engine_free(self.handle)
            self.handle = None

class W2lDecoder:
    def __init__(self, encoder, lm, lexicon, lexicon_flat):
        if not os.path.exists(lm):
            raise FileNotFoundError(lm)
        if not os.path.exists(lexicon):
            raise FileNotFoundError(lexicon)
        if not os.path.exists(lexicon_flat):
            lib.w2l_make_flattrie(encoder.tokens.encode('utf8'), lm.encode('utf8'),
                                  lexicon.encode('utf8'), lexicon_flat.encode('utf8'))

        decode_opts = ffi.new('w2l_decode_options *')
        decode_opts.beamsize = 250
        decode_opts.beamthresh = 23.530
        decode_opts.lmweight = 2.1
        decode_opts.wordscore = 3.41
        decode_opts.unkweight = -float('Inf')
        decode_opts.logadd = False
        decode_opts.silweight = 0.04
        self.decode_opts = decode_opts

        self.encoder = encoder
        self.lm = lm
        self.lexicon = lexicon
        self.lexicon_flat = lexicon_flat
        self.handle = None
        self.handle = lib.w2l_decoder_new(encoder.handle, lm.encode('utf8'),
                                          lexicon.encode('utf8'), lexicon_flat.encode('utf8'), decode_opts)

    def decode(self, samples):
        array = ffi.new('float []', list(samples))
        emission = lib.w2l_engine_process(self.encoder.handle, array, len(array))
        decode_result = lib.w2l_decoder_decode(self.handle, emission)
        decode_text = lib.w2l_decoder_result_words(self.handle, decode_result)
        decoded = consume_c_text(decode_text, sep=' ')
        lib.w2l_decoderesult_free(decode_result)
        lib.w2l_emission_free(emission)
        return ' '.join(decoded)

    def __del__(self):
        if self.handle:
            lib.w2l_decoder_free(self.handle)
            self.handle = None

class W2lLoader:
    def __init__(self, path):
        load_w2l(path)
        self.path = path
        tokens = os.path.join(path, 'tokens.txt')
        if not os.path.exists(tokens):
            raise FileNotFoundError(tokens)
        with open(tokens, 'r') as f:
            tokens = f.read().strip().split('\n')
        alphabet = os.path.join(path, 'alphabet.txt')
        if not os.path.exists(alphabet):
            with open(alphabet, 'w') as f:
                for token in tokens:
                    if token == '|': token = ' '
                    f.write(token + '\n')

    def load_model(self, models, alphabet, lm, lexicon, trie):
        am = os.path.join(self.path, 'acoustic.bin')
        tokens = os.path.join(self.path, 'tokens.txt')
        encoder = W2lEncoder(am, tokens)
        decoder = W2lDecoder(encoder, lm, lexicon, lexicon + '.flat')
        return decoder

    def stt(self, decoder, audio, fs):
        if fs != 16000:
            raise RuntimeError('only 16khz sample rate is supported')
        return decoder.decode(audio)

    def resolve_models(self, dir_name):
        am = os.path.join(self.path, 'acoustic.bin')
        lm = os.path.join(self.path, 'lm-ngram.bin')
        return am, lm, None
