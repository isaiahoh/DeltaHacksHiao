# Sets the window size
from kivy.config import Config
Config.set('graphics', 'position', 'custom')
Config.set('graphics', 'left', 20)
Config.set('graphics', 'top',  40)
Config.set('graphics', 'resizable', 0)

from kivy.core.window import Window
Window.size = (1280, 720)

from kivy.app import App
from kivy.lang import Builder
from kivy.uix.widget import Widget
from kivy.config import Config
from kivy.uix.screenmanager import ScreenManager, Screen, NoTransition
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image


class InitiationScreen(Screen):
    pass
class OriginalScreen(Screen):
    pass
class DiscernedScreen(Screen):
    pass
class MaskScreen(Screen):
    pass
class ScreenManagement(ScreenManager):
    pass

presentation = Builder.load_file("imagedisplay.kv")

class windowElements(BoxLayout):
    pass

#sets the window and the elemnts that will be run
class mainWindow(App):
    def build(self):
        Window.bind(on_dropfile=self._on_file_drop)
        return presentation

    def _on_file_drop(self, window, file_path):
        self.root.current = "original"
        return file_path
# initiates the running of the program
main = mainWindow()
main.run()