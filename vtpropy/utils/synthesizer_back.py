"""
@author: Juan Manuel Acevedo Valle
@created: Juen 2017
"""
import matplotlib
from scipy.integrate import odeint
from scipy import linspace
import ipywidgets as widgets
from ipywidgets import Layout
from divapy import Diva
import bqplot.pyplot as plt
from ipywidgets import HBox, VBox, Box, jslink
import numpy as np
import pickle

# try:
#    from PySide import QtCore, QtGui
# except ImportError:
from PyQt4 import QtCore, QtGui




from bqplot import ( LinearScale, ColorAxis,
    Axis, Scatter, Lines, CATEGORY10, Label, Figure, Tooltip
)


default_params = {'x0': np.array([0.0]*13),
                  'xp0': np.array([0.0]*13), 'm': np.array([0.0] * 13),
                  'w0': 2 * np.pi / 0.4, 'damping_factor': 1.01, 'D_t': 0.4}

# matplotlib.rcParams['figure.figsize'] = (5.5, 5.0)
# matplotlib.rcParams.update({'font.size': 9})
# matplotlib.rcParams['xtick.major.pad'] = 8
# matplotlib.rcParams['ytick.major.pad'] = 8

class Articulation(object):
    def __init__(self, **kargs):
        default_params_ = {'x0': np.array([0.0] * 13),
                          'xp0': np.array([0.0] * 13), 'm': np.array([0.0] * 13),
                          'w0': 2 * np.pi / 0.4, 'damping_factor': 1.01, 'D_t': 0.4}
        self.params = default_params_
        for key in kargs.keys():
            self.params[key] = kargs[key]

    def defaul(self):
        # params = {'x0': 0.0, 'm': np.array([0.0] * 13),'w0': 2 * np.pi / 0.4, 'damping_factor':1.01, 'D_t': 0.4}
        self.params = default_params

sys = Diva()
sys.af_play = np.array([0.]*5)
sys.outline_play = [np.array([0.]*5)]*5
sys.articulations = []
sys.articulatory_x0 = np.zeros((13,))
sys.articulatory_xp0 = np.zeros((13,))

m = []
for i in range(10):
    m += [widgets.FloatSlider(value=0, min=-3., max=3.,
                              step=0.01,
                              disabled=False,
                              continuous_update=False,
                              description='M' + str(i),
                              orientation='vertical',
                              readout=True,
                              readout_format='f',
                              slider_color='white'
                              )]

for i in range(3):
    m += [widgets.FloatSlider(value=0.2, min=0., max=1.,
                              step=0.01,
                              disabled=False,
                              continuous_update=False,
                              description='M' + str(i + 10),
                              orientation='vertical',
                              readout=True,
                              readout_format='f',
                              slider_color='white'
                              )]

params_box = []
params_box += [widgets.Label(value='$$\Delta t$$', layout=Layout(width='3%'))]
params_box += [widgets.FloatSlider(value = default_params['D_t'],
                                   min=0.01, max = 1, step = 0.01,
                                   orientation='horizontal',layout=Layout(width='20%'))]


params_box += [widgets.Label(value='$$\omega _0$$', layout=Layout(width='3%'))]
params_box += [widgets.FloatSlider(value=default_params['w0'],
                                   min=0.5, max = 30, step = 0.5,
                                   orientation='horizontal',layout=Layout(width='20%'))]

params_box += [widgets.Label(value='$$\zeta $$', layout=Layout(width='3%'))]
params_box += [widgets.FloatSlider(value=default_params['damping_factor'],
                                   min=0.5, max = 2, step = 0.02,
                                   orientation='horizontal',layout=Layout(width='20%'))]

add_btn = widgets.Button(description="Set x_0", disabled=False,layout=Layout(width='12%'))
play_all_btn = widgets.Button(description="Play all", disabled=False,layout=Layout(width='12%'))

sound_chk = widgets.Checkbox(value=True, description="Play sound:", layout=Layout(width='12%'))
sound_btn = widgets.Button(description="Repeat", disabled=True, layout=Layout(width='10%'))
art_slt = widgets.Select(options=[],
                         value=None,
                         # rows=10,
                         description='Art:',
                         disabled=False, layout=Layout(width='13%'))
delete_btn = widgets.Button(description="Remove", disabled=False, layout=Layout(width='10%'))
replace_btn = widgets.Button(description="Replace", disabled=True, layout=Layout(width='10%'))

save_btn = widgets.Button(description="Save",disable=False, layout=Layout(width='10%'))
open_btn = widgets.Button(description="Open",disable=False, layout=Layout(width='10%'))

file_box = Box([save_btn, open_btn])
file_box.layout.display = 'flex'
file_box.layout.justify_content = 'flex-end'
file_box.layout.align_itmes = 'stretch'

control_box1 = VBox([HBox(m),HBox(params_box+[sound_chk, sound_btn]),
                     HBox([add_btn, art_slt, delete_btn, replace_btn, play_all_btn])])


fig_margin_default = {'top':40, 'bottom':40, 'left':40, 'right':40}
min_height_default = 10
min_width_default = 10
# Create Sound Wave plot container
x_time = LinearScale(min=0., max=100)
y_sound = LinearScale(min=-.5, max=.5)
ax_sound_y = Axis(label='Amplitude', scale=y_sound, orientation='vertical', side='left', grid_lines='solid')
ax_time_x = Axis(label='Time', scale=x_time, grid_lines='solid')
#Initialization
sound_line = Lines(x=[], y=[], colors=['Blue'],
                       scales={'x': x_time, 'y': y_sound}, visible=True)
fig_sound = plt.figure(marks=[sound_line], axes=[ax_time_x, ax_sound_y], title='Sound wave', fig_margin = fig_margin_default,
                    min_height = min_height_default, min_widht = min_width_default, preserve_aspect=True)

# Create Articulator Position Evolution Container
y_art = LinearScale(min=-3., max=3.)
ax_art_y = Axis(label='Position', scale=y_art, orientation='vertical', side='left', grid_lines='solid')
# ax_time_x = Axis(label='Time', scale=x_time, grid_lines='solid')
#Initialization
art_lines = Lines(x=[], y=[],  #colors=['Blue'],
                       scales={'x': x_time, 'y': y_art}, visible=True)
fig_art = plt.figure(marks=[art_lines], axes=[ax_time_x, ax_art_y], title='Articulators', fig_margin = fig_margin_default,
                    min_height = min_height_default, min_widht = min_width_default)


# Create Formants Evolution Container
y_formant = LinearScale(min=0., max=3200.)
ax_formant_y = Axis(label='Frequency', scale=y_formant, orientation='vertical', side='left', grid_lines='solid')
# ax_time_x = Axis(label='Time', scale=x_time, grid_lines='solid')
#Initialization
formant_lines = Lines(x=[], y=[],  #colors=['Blue'],
                       scales={'x': x_time, 'y': y_formant}, visible=True)
fig_formant = plt.figure(marks=[formant_lines], axes=[ax_time_x, ax_formant_y], title='Formant Frequencies', fig_margin = fig_margin_default,
                    min_height = min_height_default, min_widht = min_width_default)

# Create Somatic Evolution Container
y_somatic = LinearScale(min=-1., max=1.)
ax_somatic_y = Axis(label='Value', scale=y_somatic, orientation='vertical', side='left', grid_lines='solid')
# ax_time_x = Axis(label='Time', scale=x_time, grid_lines='solid')
#Initialization
# somatic = [np.random.rand(100) for x in range(8)]
somatic_lines = Lines(x=[], y=[],  #colors=['Blue'],
                       scales={'x': x_time, 'y': y_somatic}, visible=True)
fig_somatic = plt.figure(marks=[somatic_lines], axes=[ax_time_x, ax_somatic_y], title='Somatic signals', fig_margin = fig_margin_default,
                    min_height = min_height_default, min_widht = min_width_default)

# Create animated controls
play = widgets.Play(
        # interval=0.005,
        value=0,
        min=0,
        max=40,
        step=20,
        description="Press play",
        disabled=False)
slider = widgets.IntSlider(min = 0, max = 40, value=0)
widgets.jslink((play, 'value'), (slider, 'value'))

step_txt = widgets.IntText(value=20, min=1, max=40, description="Step", layout=Layout(width='12%'))

# Create animated vocal tract container
x_vt = LinearScale(min=-50., max=200)
y_vt = LinearScale(min=-200., max=100.)
ax_vt_y = Axis(label='', scale=y_vt, orientation='vertical', side='left', grid_lines='none', visible=False)
ax_vt_x = Axis(label='', scale=x_vt, grid_lines='none', visible=False)
vt_line1 = Lines(x=[], y=[],scales={'x': x_vt, 'y': y_vt}, visible=True)
vt_line2 = Lines(x=[], y=[],scales={'x': x_vt, 'y': y_vt}, visible=True)
vt_line3 = Lines(x=[], y=[],scales={'x': x_vt, 'y': y_vt}, visible=True)
fig_outline = plt.figure(marks=[vt_line1, vt_line2, vt_line3], axes=[ax_vt_x, ax_vt_y],
                         title='Vocal Tract Shape', fig_margin = fig_margin_default,
                    min_height = min_height_default, min_widht = min_width_default, preserve_aspect=True,
                         animation_duration=0)

# create animated areafunction container
x_af = LinearScale(min=0., max=220)
y_af = LinearScale(min=-1., max=3.)
ax_af_y = Axis(label='Lenght', scale=y_af, orientation='vertical', side='left', grid_lines='solid')
ax_af_x = Axis(label='Position', scale=x_af, grid_lines='solid')
af_line = Lines(x=[], y=[],scales={'x': x_af, 'y': y_af}, visible=True)
fig_af = plt.figure(marks=[af_line], axes=[ax_af_x, ax_af_y], title='Area Function', fig_margin = fig_margin_default,
                    min_height = min_height_default, min_widht = min_width_default,
                         animation_duration=0)


plots_box1 = HBox([fig_outline, fig_af, fig_sound],layout=Layout(width='99%', height='310px'))#])
plots_box2 = HBox([fig_art, fig_formant, fig_somatic],layout=Layout(width='99%', height='310px'))

def update_vocalization():
    art_traj = get_art_trajectory()
    aud, som, outline, af = sys.get_audsom(art_traj)
    sys.af_play = af
    sys.outline_play = outline
    sound = sys.get_sound(art_traj)
    n_s = aud.shape[0]
    max_time = (n_s-1)*0.005
    sys.max_time_play =  max_time
    sys.time_play = np.linspace(0, max_time, n_s)
    x_time.max = max_time
    sys.art_traj_play  = art_traj
    sys.aud_play = aud
    sys.som_play = som
    sys.sound_play = sound

def update_plots():
    max_time = sys.max_time_play
    sound = sys.sound_play
    time = sys.time_play
    art_traj = sys.art_traj_play
    aud = sys.aud_play
    som = sys.som_play
    n_s = aud.shape[0]

    # Update Sound wave plot
    sound_time = np.linspace(0,max_time,len(sound))
    y_sound.min=min(y_sound.min, min(sound))
    y_sound.max=max(y_sound.max, max(sound))
    sound_line.y = sound
    sound_line.x = sound_time
    # sound_line.visible = True

    # Update articulators' position plot
    art_lines.x = time
    art_lines.y = art_traj.transpose()
    # art_lines.visible =True

    # Update formants' plot
    formant_lines.x = time
    formant_lines.y = aud.transpose()
    # formant_lines.visible =True

    # Update somatic signils' plot
    somatic_lines.x = time
    somatic_lines.y = som.transpose()
    # somatic_lines.visible = True

    # slider.value = 0
    idx = n_s - 1
    slider.max = idx
    play.max = n_s -1
    slider.value = idx

    af_line.x = range(len(sys.af_play[idx]))
    af_line.y = sys.af_play[idx]

    vt_line1.x = np.real(sys.outline_play[idx][:352])
    vt_line1.y = np.imag(sys.outline_play[idx][:352])
    vt_line2.x = np.real(sys.outline_play[idx][352:354])
    vt_line2.y = np.imag(sys.outline_play[idx][352:354])
    vt_line3.x = np.real(sys.outline_play[idx][354:])
    vt_line3.y = np.imag(sys.outline_play[idx][354:])

def step_txt_callback(foo):
    step = step_txt.get_interact_value()
    play.step = step

def slider_change_callback(foo):
    idx =  slider.get_interact_value()

    af_line.x = range(len(sys.af_play[idx]))
    x_af.min = 0
    x_af.max = len(sys.af_play[idx][2:-2])
    af_line.y = sys.af_play[idx][2:-2]

    vt_line1.x = np.real(sys.outline_play[idx][:352])
    vt_line1.y = np.imag(sys.outline_play[idx][:352])
    vt_line2.x = np.real(sys.outline_play[idx][352:354])
    vt_line2.y = np.imag(sys.outline_play[idx][352:354])
    vt_line3.x = np.real(sys.outline_play[idx][354:])
    vt_line3.y = np.imag(sys.outline_play[idx][354:])

    print(play.get_state('_playing')['_playing'])
    print(slider.value + step_txt.value)

    if play.get_state('_playing')['_playing'] and slider.value + step_txt.value + 1 > slider.max:
        if sound_chk.value:
            sound_btn.disabled = False
            sys.play_sound(sys.sound_play)
        play.set_trait('_playing',False)
        slider.value = slider.max


def param_change_callback(foo):
    update_vocalization()
    update_plots()
    if sound_chk.get_interact_value():
        sys.play_sound(sys.sound_play)

def play_callback(foo):
    if sound_chk.get_interact_value():
        sys.play_sound(sys.sound_play)
    slider.value=slider.max


def btn_repeat_sound_callback(foo):
    #     sound_btn.disabled =True
    if sound_chk.value:
        sound_btn.disabled = False
        sys.play_sound(sys.sound_play)

def play_all_callback(foo):
    pass



def add_articulation_callback(foo):
    if add_btn.description == "Set x_0":
        sys.articulatory_x0 = np.array([float(x.get_interact_value()) for x in m])
        sys.articulatory_xp0 = np.zeros((13,))
        add_btn.description = "Add"
    else:
        m_ = np.array([float(x.get_interact_value()) for x in m])
        D_t = params_box[1].get_interact_value()
        w0 = params_box[3].get_interact_value()
        damping_factor = params_box[5].get_interact_value()
        if len(sys.articulations)>0:
            sys.articulations += [Articulation(**{'m':m_,'w0':w0,'D_t':D_t,
                                                  'damping_factor':damping_factor})]
        else:
            sys.articulations += [Articulation(**{'m': m_, 'w0': w0, 'D_t': D_t,
                             'x0':sys.articulatory_x0, 'xp0':sys.articulatory_xp0,
                             'damping_factor': damping_factor})]
        art_slt.options = range(len(sys.articulations))
        art_slt.value = len(sys.articulations)-1

def remove_art_callback(foo):
    idx = art_slt.get_interact_value()
    del(sys.articulations[idx])
    art_slt.options = range(len(sys.articulations))
    art_slt.value = len(sys.articulations) - 1

def replace_art_callback(foo):
    pass

def save_callback(foo):
    file_name = gui_fname(mode='save')
    save_obj(sys.articulations, file_name)

def open_callback(foo):
    file_name = gui_fname(mode='open')
    sys.articulations = load_obj(file_name)

def get_art_trajectory():
    m_ = np.array([float(x.get_interact_value()) for x in m])
    D_t = params_box[1].get_interact_value()
    w0 = params_box[3].get_interact_value()
    damping_factor = params_box[5].get_interact_value()

    art_ = [Articulation(**{'m':m_,'w0':w0,'D_t':D_t, 'damping_factor':damping_factor})]

    if len(sys.articulations) > 0:
        art__ = sys.articulations[-1]
        last_motor = get_motor_dynamics([art__])
        art_[-1].params['x0'] = last_motor[-1, :13]
        art_[-1].params['xp0'] = last_motor[-1, 13:]

    arts_ = sys.articulations + art_
    motor_trajectory = get_motor_dynamics(arts_)
    return motor_trajectory[:,:13]


def get_motor_dynamics(articulations):
    ts = 0.005  # Sampling time diva vocal tract sound waves

    arts_trajectory = []
    for art_ in articulations:
        if len(arts_trajectory)==0:
            x0 = art_.params['x0']
            xp0 = art_.params['xp0']
        else:
            x0 = arts_trajectory[-1][-1,:13]
            xp0 = arts_trajectory[-1][-1,13:]
        y0 = np.zeros((26,))
        y0[:13] = x0
        y0[13:] = xp0
        m_ = art_.params['m']
        D_t = art_.params['D_t']
        w0 = art_.params['w0']
        damping_factor = art_.params['damping_factor']

        n_samples = int(D_t / ts + 1)
        y_neutral = [0.0] * 13
        y_neutral[11] = -0.25
        y_neutral[12] = -0.25
        t = linspace(0.0, D_t, n_samples)
        arts_trajectory += [odeint(motor_dynamics, y0, t, args=(m_, damping_factor, w0))]
    total_dims = sum([x.shape[0] for x in arts_trajectory])
    arts_trajectory_ = np.zeros((total_dims,26))
    filled = 0
    for x in arts_trajectory:
        to_fill = x.shape[0]
        arts_trajectory_[filled:filled+to_fill, :] = x
        filled += to_fill
    return arts_trajectory_

def motor_dynamics(y, t, m, damping_factor, w0):
    dy1 = y[13]
    dy2 = y[14]
    dy3 = y[15]
    dy4 = y[16]
    dy5 = y[17]
    dy6 = y[18]
    dy7 = y[19]
    dy8 = y[20]
    dy9 = y[21]
    dy10 = y[22]
    dy11 = y[23]
    dy12 = y[24]
    dy13 = y[25]

    dy14 = -2 * damping_factor * w0 * y[13] - (pow(w0, 2)) * y[0] + (pow(w0, 2)) * m[0]
    dy15 = -2 * damping_factor * w0 * y[14] - (pow(w0, 2)) * y[1] + (pow(w0, 2)) * m[1]
    dy16 = -2 * damping_factor * w0 * y[15] - (pow(w0, 2)) * y[2] + (pow(w0, 2)) * m[2]
    dy17 = -2 * damping_factor * w0 * y[16] - (pow(w0, 2)) * y[3] + (pow(w0, 2)) * m[3]
    dy18 = -2 * damping_factor * w0 * y[17] - (pow(w0, 2)) * y[4] + (pow(w0, 2)) * m[4]
    dy19 = -2 * damping_factor * w0 * y[18] - (pow(w0, 2)) * y[5] + (pow(w0, 2)) * m[5]
    dy20 = -2 * damping_factor * w0 * y[19] - (pow(w0, 2)) * y[6] + (pow(w0, 2)) * m[6]
    dy21 = -2 * damping_factor * w0 * y[20] - (pow(w0, 2)) * y[7] + (pow(w0, 2)) * m[7]
    dy22 = -2 * damping_factor * w0 * y[21] - (pow(w0, 2)) * y[8] + (pow(w0, 2)) * m[8]
    dy23 = -2 * damping_factor * w0 * y[22] - (pow(w0, 2)) * y[9] + (pow(w0, 2)) * m[9]
    dy24 = -2 * damping_factor * w0 * y[23] - (pow(w0, 2)) * y[10] + (pow(w0, 2)) * m[10]
    dy25 = -2 * damping_factor * w0 * y[24] - (pow(w0, 2)) * y[11] + (pow(w0, 2)) * m[11]
    dy26 = -2 * damping_factor * w0 * y[25] - (pow(w0, 2)) * y[12] + (pow(w0, 2)) * m[12]

    return [dy1, dy2, dy3, dy4, dy5, dy6, dy7, dy8, dy9, dy10, dy11, dy12, dy13, dy14, dy15, dy16, dy17, dy18, dy19,
            dy20, dy21, dy22, dy23, dy24, dy25, dy26]


def save_obj(obj, name ):
    with open(''+ name, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open('' + name, 'rb') as f:
        return pickle.load(f)

def gui_fname(dir=None, mode='open'):
    """Select a file via a dialog and returns the file name.
    """
    if dir is None: dir = './'

    if mode=='open':
        fname = QtGui.QFileDialog.getOpenFileName(None, "Select data file...",
                                              dir, filter="All files (*);; SM Files (*.sm)")
    elif mode=='save':
        fname = QtGui.QFileDialog.getSaveFileName(None, "Select data file...",
                                              dir, filter="All files (*);; SM Files (*.sm)")
    try:
        return fname
    except IndexError:
        return None