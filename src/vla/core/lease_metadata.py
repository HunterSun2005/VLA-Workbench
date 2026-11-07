"""Release diagnostic owner records without removing flock's stable inode."""


def release_owned_streams(streams):
    error = None
    while streams:
        stream = streams.pop()
        try:
            # Only call with an acquired exclusive lease, before closing/unlocking.
            stream.seek(0)
            stream.truncate()
            stream.flush()
        except Exception as exc:
            error = error or exc
        finally:
            try:
                stream.close()
            except Exception as exc:
                error = error or exc
    if error is not None:
        raise error
