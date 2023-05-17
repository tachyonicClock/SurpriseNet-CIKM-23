from dotenv import load_dotenv
from sshtunnel import SSHTunnelForwarder
from vizier.service import clients
from vizier.service import pyvizier as vz
import os
import typing as t

from absl.flags import FLAGS

FLAGS([""])
load_dotenv()

VIZIER_HOST = os.environ.get("VIZIER_HOST", None)
VIZIER_PORT = os.environ.get("VIZIER_PORT", None)


def _via_ssh(
    vizier_host: t.Optional[str],
    vizier_port: t.Union[int, str, None],
):
    if vizier_port is None or vizier_host is None:
        raise ValueError(
            "Vizier port or host must be specified either as an argument or as "
            + "an environment variable. Use the `VIZIER_HOST` and `VIZIER_PORT`"
        )
    vizier_port = int(vizier_port)

    server = SSHTunnelForwarder(
        vizier_host,
        remote_bind_address=("localhost", vizier_port),
    )
    server.start()
    address, port = server.local_bind_address
    clients.environment_variables.server_endpoint = f"{address}:{port}"


def vizier_client_via_ssh(
    owner: str,
    study_id: str,
    study_config: t.Optional[vz.StudyConfig] = None,
    vizier_host: t.Optional[str] = VIZIER_HOST,
    vizier_port: t.Union[int, str, None] = VIZIER_PORT,
) -> clients.Study:
    """Create a Vizier client via SSH.

    :param study_config: A Vizier study configuration.
    :param owner: The entity responsible for the study, used in the resource identifier.
    :param study_id: The study identifier, used in the resource identifier.
    :param vizier_host: The host of the Vizier server.
    :param vizier_port: The port of the Vizier server to be forwarded.
    :return: A Vizier client for the study.
    """
    _via_ssh(vizier_host, vizier_port)
    if study_config is None:
        return clients.Study.from_owner_and_id(owner=owner, study_id=study_id)
    else:
        return clients.Study.from_study_config(
            study_config, owner=owner, study_id=study_id
        )
