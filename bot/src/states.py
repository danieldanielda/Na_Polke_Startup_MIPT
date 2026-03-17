from aiogram.fsm.state import StatesGroup, State

class UserState(StatesGroup):
    waiting_for_photo = State()
    waiting_for_analysis_type = State()
    analysis_finished = State()