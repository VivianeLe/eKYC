import streamlit as st
from streamlit.runtime.scriptrunner import get_script_run_ctx
from streamlit.runtime.state.session_state import SessionState as BaseSessionState


class SessionState(BaseSessionState):
    def __init__(self, **kwargs):
        """A new SessionState object.

        Parameters
        ----------
        **kwargs : any
            Default values for the session state.

        Example
        -------
        >>> session_state = SessionState(user_name='', favorite_color='black')
        >>> session_state.user_name = 'Mary'
        ''
        >>> session_state.favorite_color
        'black'

        """
        for key, val in kwargs.items():
            setattr(self, key, val)


def get(**kwargs):
    """Gets a SessionState object for the current session.

    Creates a new object if necessary.

    Parameters
    ----------
    **kwargs : any
        Default values you want to add to the session state, if we're creating a
        new one.

    Example
    -------
    >>> session_state = get(user_name='', favorite_color='black')
    >>> session_state.user_name
    ''
    >>> session_state.user_name = 'Mary'
    >>> session_state.favorite_color
    'black'

    Since you set user_name above, next time your script runs this will be the
    result:
    >>> session_state = get(user_name='', favorite_color='black')
    >>> session_state.user_name
    'Mary'

    """
    # Hack to get the session object from Streamlit.

    ctx = get_script_run_ctx()

    if ctx is None:
        raise RuntimeError("Couldn't get your Streamlit ScriptRunContext. Are you running outside of Streamlit?")

    session_id = ctx.session_id
    current_session = st.session_state.get("_custom_session_state", None)

    if current_session is None:
        # Create a new session state if it doesn't exist
        current_session = SessionState(**kwargs)
        st.session_state["_custom_session_state"] = current_session

    return current_session
